#!/usr/bin/env python3
"""Run a pedestrian detector on recorded LiDAR or RealSense data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def mid360_points(path, intensity_scale):
    flat = np.fromfile(path, dtype="<f4")
    if flat.size % 4 or path.stat().st_size % 16:
        raise ValueError(f"{path}: expected packed float32 XYZI")
    points = flat.reshape(-1, 4)
    points = points[np.isfinite(points).all(axis=1)]
    points[:, 3] *= intensity_scale
    return points


def realsense_points(depth, calibration):
    """Unproject rectified optical depth and transform to detector input axes."""
    fx, fy, cx, cy, scale = (
        float(calibration[key]) for key in ("fx", "fy", "cx", "cy", "depth_scale_m")
    )
    rotation = np.asarray(calibration["optical_to_input"], dtype=np.float64)
    if (
        depth.ndim != 2
        or not np.isfinite([fx, fy, cx, cy, scale]).all()
        or min(fx, fy, scale) <= 0
        or rotation.shape != (4, 4)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation[3], [0, 0, 0, 1])
        or not np.allclose(rotation[:3, :3].T @ rotation[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation[:3, :3]), 1.0)
    ):
        raise ValueError("expected rectified 2D depth, valid intrinsics and a rigid optical_to_input")
    z_m = depth.astype(np.float64) * scale
    v, u = np.nonzero(np.isfinite(z_m) & (z_m > 0))
    z_m = z_m[v, u]
    optical = np.column_stack(((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m))
    xyz = optical @ rotation[:3, :3].T + rotation[:3, 3]
    return np.column_stack((xyz, np.zeros(len(xyz)))).astype(np.float32)


def pedestrian_detections(boxes, scores, labels, pedestrian_id, score_min, z_offset_m):
    """Translate OpenPCDet boxes into tracker class 2 detections."""
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError("prediction arrays differ in length")
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if label != pedestrian_id or not np.isfinite(score) or not score_min <= score <= 1:
            continue
        center = np.asarray(box[:3], dtype=np.float64).copy()
        dimensions = np.asarray(box[3:6], dtype=np.float64)
        yaw = float(box[6])
        if not np.isfinite(box[:7]).all() or np.any(dimensions <= 0):
            continue
        center[2] -= z_offset_m
        signs = np.array([[x, y, z] for z in (-1, 1)
                          for x, y in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        corners = signs * dimensions / 2
        cos, sin = math.cos(yaw), math.sin(yaw)
        corners[:, :2] = corners[:, :2] @ np.array([[cos, sin], [-sin, cos]])
        length, width, height = dimensions.tolist()
        detections.append({
            "position": center.tolist(),
            "covariance": (np.eye(3) * 0.04).ravel().tolist(),
            "class": 2,
            "score": float(score),
            "shape": {"height": height, "width": width, "length": length,
                      "rotation_y": 0.0, "corners": (corners + center).ravel().tolist()},
        })
    return detections


def replace_detections(template, output, lidar_sensor_id, read_points, predict):
    """Replace one sensor's detections while preserving timestamps and other records."""
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with template.open() as source, NamedTemporaryFile(
            mode="w", dir=output.parent, delete=False,
        ) as destination:
            temp_path = Path(destination.name)
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["type"] == "lidar_batch" and record["sensor_id"] == lidar_sensor_id:
                    record["detections"] = predict(read_points(record["sequence"]))
                    count += 1
                destination.write(json.dumps(record, allow_nan=False) + "\n")
        if count == 0:
            raise ValueError(f"no lidar_batch records for sensor {lidar_sensor_id}")
        temp_path.replace(output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return count


def openpcdet_predictor(args):
    try:
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
    except ImportError as error:
        raise RuntimeError("run inference in an OpenPCDet environment with CUDA") from error
    if not torch.cuda.is_available():
        raise RuntimeError("OpenPCDet inference requires a CUDA device")

    cfg_from_yaml_file(args.cfg, cfg)
    pedestrian_id = list(cfg.CLASS_NAMES).index("Pedestrian") + 1
    logger = common_utils.create_logger()
    dataset = DatasetTemplate(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        training=False, root_path=None, logger=logger,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.ckpt), logger=logger, to_cpu=True)
    model.cuda().eval()

    def predict(points):
        if not len(points):
            return []
        shifted = points.copy()
        shifted[:, 2] += args.z_offset_m
        batch = dataset.collate_batch([dataset.prepare_data({"points": shifted, "frame_id": 0})])
        if not len(batch["voxels"]):
            return []
        load_data_to_gpu(batch)
        with torch.no_grad():
            predictions, _ = model.forward(batch)
        result = predictions[0]
        boxes, scores, labels = (result[key].cpu().numpy() for key in
                                 ("pred_boxes", "pred_scores", "pred_labels"))
        return pedestrian_detections(boxes, scores, labels, pedestrian_id, args.score_min,
                                     args.z_offset_m)

    return predict


def coda_cpu_predictor(args):
    """Use the existing CPU implementation of the CODa first-stage models."""
    module_path = Path(__file__).resolve().parents[1] / "coda" / "detect_lidar_cpu.py"
    spec = importlib.util.spec_from_file_location("detect_lidar_cpu", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        raise RuntimeError("CODa CPU inference requires PyTorch and Shapely") from error

    module.torch.set_num_threads(args.cpu_threads)
    checkpoint = module.torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state = {key: value.float() for key, value in checkpoint["model_state"].items()
             if value.is_floating_point()}
    class_names, point_range = module.model_config(state)
    pedestrian_id = class_names.index("Pedestrian")

    def predict(points):
        shifted = points.copy()
        shifted[:, 2] += args.z_offset_m
        # The CODa model itself applies an additional training-time shift.
        grid_points = shifted[:, :3] + [0, 0, module.SHIFT_Z]
        inside = np.all((grid_points >= point_range[:3]) & (grid_points < point_range[3:]), axis=1)
        if not inside.any():
            return []
        boxes, scores, labels = module.detect(shifted, state, args.score_min, args.nms_threshold)
        return pedestrian_detections(boxes, scores, labels, pedestrian_id, args.score_min,
                                     args.z_offset_m)

    return predict


def cluster_predictor(args):
    """Person-size geometric baseline using example 151's scan-only rules."""
    try:
        from scipy.ndimage import label
    except ImportError as error:
        raise RuntimeError("cluster baseline requires SciPy") from error

    cell_m = 0.15
    range_m = 15.0
    half_grid = math.ceil(range_m / cell_m)

    def predict(points):
        height_m = points[:, 2] - args.floor_z_m
        accepted = ((height_m >= 0.15) & (height_m <= 1.9)
                    & (np.linalg.norm(points[:, :2], axis=1) < range_m))
        xyz = points[accepted, :3]
        if not len(xyz):
            return []
        cells = np.floor(xyz[:, :2] / cell_m).astype(int) + half_grid
        occupied = np.zeros((2 * half_grid, 2 * half_grid), dtype=bool)
        occupied[cells[:, 0], cells[:, 1]] = True
        labels, _ = label(occupied, structure=np.ones((3, 3), dtype=bool))
        point_labels = labels[cells[:, 0], cells[:, 1]]
        order = np.argsort(point_labels, kind="stable")
        boundaries = np.flatnonzero(np.diff(point_labels[order])) + 1
        boxes = []
        for indices in np.split(order, boundaries):
            if len(indices) < 4:
                continue
            group = xyz[indices]
            lo, hi = group.min(axis=0), group.max(axis=0)
            size = hi - lo
            if np.linalg.norm(size[:2]) > 1.0 or size[2] < 0.2:
                continue
            boxes.append([*((lo + hi) / 2), *np.maximum(size, cell_m), 0.0])
        return pedestrian_detections(boxes, [1.0] * len(boxes), [1] * len(boxes),
                                     1, args.score_min, 0.0)

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=("mid360", "xt32", "realsense"), required=True)
    parser.add_argument("--scans", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("openpcdet", "coda-cpu", "cluster"), default="openpcdet")
    parser.add_argument("--cfg", help="matching OpenPCDet config")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--nms-threshold", type=float, default=0.3,
                        help="CODa CPU rotated BEV NMS threshold")
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--floor-z-m", type=float, help="cluster floor height in the input frame")
    parser.add_argument("--calibration", type=Path, help="rectified RealSense intrinsics and mount")
    parser.add_argument("--lidar-sensor-id", type=int, default=1)
    parser.add_argument("--score-min", type=float, default=0.3)
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--z-offset-m", type=float, default=0.0)
    args = parser.parse_args()
    if args.backend == "cluster":
        if args.floor_z_m is None or not math.isfinite(args.floor_z_m):
            parser.error("--floor-z-m is required and must be finite for the cluster backend")
    elif args.ckpt is None:
        parser.error("--ckpt is required for learned detectors")
    if args.backend == "openpcdet" and args.cfg is None:
        parser.error("--cfg is required with --backend openpcdet")
    if args.cpu_threads < 1 or not 0 <= args.nms_threshold <= 1:
        parser.error("CPU thread count must be positive and NMS threshold must be in [0, 1]")
    if not 0 <= args.score_min <= 1 or not math.isfinite(args.z_offset_m):
        parser.error("score must be in [0, 1] and z offset must be finite")
    if not math.isfinite(args.intensity_scale) or args.intensity_scale <= 0:
        parser.error("intensity scale must be finite and positive")
    if args.template.resolve() == args.out.resolve():
        parser.error("choose an output path different from the template")
    if args.sensor == "realsense":
        if args.calibration is None:
            parser.error("--calibration is required for RealSense")
        calibration = json.loads(args.calibration.read_text())

        def read_points(sequence):
            depth = np.load(args.scans / f"{sequence:06d}.npy", allow_pickle=False)
            return realsense_points(depth, calibration)
    else:
        def read_points(sequence):
            return mid360_points(args.scans / f"{sequence:06d}.bin", args.intensity_scale)

    try:
        backends = {"openpcdet": openpcdet_predictor, "coda-cpu": coda_cpu_predictor,
                    "cluster": cluster_predictor}
        predict = backends[args.backend](args)
        times_ms = []

        def timed_predict(points):
            start = time.perf_counter()
            detections = predict(points)
            times_ms.append((time.perf_counter() - start) * 1000)
            return detections

        count = replace_detections(args.template, args.out, args.lidar_sensor_id,
                                   read_points, timed_predict)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    median_ms, p95_ms = np.percentile(times_ms, [50, 95])
    print(f"{count} {args.sensor} scans -> {args.out}")
    print(f"Preprocess + inference + box export: median {median_ms:.1f} ms, p95 {p95_ms:.1f} ms")


if __name__ == "__main__":
    main()
