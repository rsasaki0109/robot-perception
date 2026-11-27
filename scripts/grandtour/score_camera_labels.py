"""Score projected ETH-2 LiDAR detections against camera-only person labels."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import zarr

from grandtour import Odometry


def project(corners, optical_from_base, intrinsics, distortion, scale):
    optical = corners @ optical_from_base[:3, :3].T + optical_from_base[:3, 3]
    if np.any(optical[:, 2] <= 0):
        return None
    xy = optical[:, :2] / optical[:, 2, None]
    radius = np.linalg.norm(xy, axis=1)
    theta = np.arctan(radius)
    angle = theta * np.polynomial.polynomial.polyval(theta**2, [1, *distortion])
    uv = xy * (angle / np.maximum(radius, 1e-12))[:, None]
    uv = (uv @ intrinsics[:2, :2].T + intrinsics[:2, 2]) * scale
    return np.r_[uv.min(axis=0), uv.max(axis=0)]


def contains(box, center):
    return bool(np.all(box[:2] <= center) and np.all(center < box[2:]))


def iou(left, right):
    intersection = np.maximum(
        0, np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2])
    ).prod()
    union = (left[2:] - left[:2]).prod() + (right[2:] - right[:2]).prod() - intersection
    return float(intersection / union)


def score(predictions, labels, ignore):
    overlaps = np.array(
        [[iou(prediction, label) for label in labels] for prediction in predictions]
    ).reshape(len(predictions), len(labels))
    eligible = overlaps >= 0.30
    # Prefer the largest number of matches, breaking ties by total IoU.
    rows, cols = linear_sum_assignment(
        eligible * (len(labels) + overlaps), maximize=True
    )
    matched = {row for row, col in zip(rows, cols) if eligible[row, col]}
    ignored = sum(
        any(contains(box, (prediction[:2] + prediction[2:]) / 2) for box in ignore)
        for row, prediction in enumerate(predictions)
        if row not in matched
    )
    return len(matched), len(predictions) - len(matched) - ignored, len(labels) - len(matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("batches", type=Path, nargs="+")
    args = parser.parse_args()
    annotation = json.loads(args.labels.read_text())
    frames = {int(index): frame for index, frame in annotation["frames"].items()}
    open_setup = gzip.open if args.setup.suffix == ".gz" else open
    with open_setup(args.setup, "rt") as stream:
        setup = json.load(stream)
    camera_from_base = np.linalg.inv(np.array(setup["camera_to_reference"]).reshape(4, 4))
    root = zarr.open_group(str(args.data), mode="r")
    camera = root[annotation["camera"]]
    info = camera.attrs["camera_info"]
    intrinsics = np.array(info["K"]).reshape(3, 3)
    scale = np.array(annotation["image_size_px"]) / [info["width"], info["height"]]
    odometry = Odometry(root["anymal_state_odometry"])
    roi = np.array(annotation["roi_xyxy_px"])
    for path in args.batches:
        counts = {}
        for line in path.read_text().splitlines():
            batch = json.loads(line)
            if batch["type"] != "lidar_batch" or batch["sequence"] not in frames:
                continue
            index = batch["sequence"]
            camera_time = float(camera["timestamp"][index])
            scan_time = batch["event_time_ns"] * 1e-9
            transform = camera_from_base @ np.linalg.solve(
                odometry.at(camera_time), odometry.at(scan_time)
            )
            predictions = []
            for detection in batch["detections"]:
                if detection["class"] != 2:
                    continue
                box = project(
                    np.array(detection["shape"]["corners"]).reshape(8, 3),
                    transform, intrinsics, info["D"], scale,
                )
                if box is not None and contains(roi, (box[:2] + box[2:]) / 2):
                    predictions.append(box)
            frame = frames[index]
            counts[index] = score(predictions, np.array(frame["persons"]), np.array(frame["ignore"]))
        if counts.keys() != frames.keys():
            raise ValueError(f"{path}: missing annotated frames {sorted(frames.keys() - counts.keys())}")
        tp, fp, fn = np.sum(list(counts.values()), axis=0).tolist()
        print(json.dumps(dict(file=str(path), tp=tp, fp=fp, fn=fn, frames=counts)))


if __name__ == "__main__":
    main()
