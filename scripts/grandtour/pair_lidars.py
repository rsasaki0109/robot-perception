#!/usr/bin/env python3
"""Prepare single-sweep ETH-2 Mid-360 and XT32 scans in their shared field of view."""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from grandtour import LIDAR, ODOMETRY, Odometry, base_to_sensor, ns

XT32 = "hesai_points_undistorted"
ELEVATION_DEG = ((-7, 52), (-16, 15))
BASE_RANGE_M = (0.5, 8.0)


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def common_mask(points, sensor_from_base):
    distance = np.linalg.norm(points[:, :2], axis=1)
    keep = (distance >= BASE_RANGE_M[0]) & (distance <= BASE_RANGE_M[1])
    for matrix, (lower, upper) in zip(sensor_from_base, ELEVATION_DEG):
        xyz = transform(points, matrix)
        elevation = np.rad2deg(np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1)))
        keep &= (elevation >= lower) & (elevation <= upper)
    return keep


def read_records(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def write_records(path, records):
    with path.open("w") as output:
        for record in records:
            output.write(json.dumps(record, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="unpacked GrandTour data Zarr")
    parser.add_argument("--template", type=Path, required=True, help="ETH-2 input JSONL")
    parser.add_argument("--truth", type=Path, required=True, help="ETH-2 truth JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-offset-ms", type=float, default=50.0)
    args = parser.parse_args()
    if not np.isfinite(args.max_offset_ms) or args.max_offset_ms <= 0:
        parser.error("--max-offset-ms must be finite and positive")

    root = zarr.open_group(str(args.data), mode="r")
    tf = root["tf"].attrs["tf"]
    groups = root[LIDAR], root[XT32]
    times = [source["timestamp"][:] for source in groups]
    mid_from_base = base_to_sensor(tf, "livox_lidar")
    xt_from_base = base_to_sensor(tf, "hesai_lidar")
    odometry = Odometry(root[ODOMETRY])
    labels = {r["timestamp_ns"]: r for r in read_records(args.truth)}

    pairs, records, cropped_labels = [], [], []
    previous_xt_scan = -1
    for record in read_records(args.template):
        if record["type"] != "lidar_batch" or record["sensor_id"] != 1:
            continue
        i = record["sequence"]
        time = times[0][i]
        if ns(time) != record["event_time_ns"] or record["frame_id"] != "base":
            raise ValueError(f"template does not match GrandTour scan {i}")
        k = int(np.clip(np.searchsorted(times[1], time), 1, len(times[1]) - 1))
        j = min((k - 1, k), key=lambda index: abs(times[1][index] - time))
        delta_ms = float((times[1][j] - time) * 1000)
        if abs(delta_ms) > args.max_offset_ms or j <= previous_xt_scan:
            continue
        previous_xt_scan = j
        world_from_base = odometry.at(time)
        world_from_xt_base = odometry.at(times[1][j])
        xt_from_mid_base = xt_from_base @ np.linalg.inv(world_from_xt_base) @ world_from_base
        sensor_from_base = (mid_from_base, xt_from_mid_base)
        objects = labels[ns(time)]
        xyz = np.asarray([o["position"] for o in objects["objects"]]).reshape(-1, 3)
        keep = common_mask(transform(xyz, np.linalg.inv(world_from_base)), sensor_from_base)
        cropped_labels.append(dict(objects, objects=[o for o, good in zip(objects["objects"], keep)
                                                      if good]))
        records.append({**record, "detections": [], "landmarks": []})
        pairs.append((i, j, delta_ms, sensor_from_base))
    if not pairs:
        raise ValueError("no one-to-one scan pairs within the timestamp tolerance")

    # Bulk reads avoid decompressing the same large Zarr chunks for every frame.
    for axis, (name, source) in enumerate(zip(("mid360", "xt32"), groups)):
        indices = [pair[axis] for pair in pairs]
        first, last = min(indices), max(indices)
        valid = source["valid"][first:last + 1, 0]
        width = int(valid.max())
        xyz = source["points"][first:last + 1, :width]
        intensity = source["intensity"][first:last + 1, :width]
        output = args.out / name
        output.mkdir(parents=True, exist_ok=True)
        for pair, scan in zip(pairs, indices):
            i = pair[0]
            sensor_from_base = pair[3]
            slot = scan - first
            count = int(valid[slot])
            points = xyz[slot, :count]
            if not count or not np.any(points):
                raise ValueError(f"missing or empty raw {name} scan {scan}")
            base_points = transform(points, np.linalg.inv(sensor_from_base[axis]))
            keep = common_mask(base_points, sensor_from_base)
            cloud = np.column_stack((base_points[keep], intensity[slot, :count][keep]))
            cloud.astype("<f4").tofile(output / f"{i:06d}.bin")

    write_records(args.out / "inputs.jsonl", records)
    write_records(args.out / "truth.jsonl", cropped_labels)
    write_records(args.out / "pairs.jsonl", [
        {"mid360_scan": i, "xt32_scan": j, "xt32_minus_mid360_ms": delta_ms}
        for i, j, delta_ms, _ in pairs
    ])
    print(f"{len(pairs)} pairs, {sum(len(r['objects']) for r in cropped_labels)} annotated objects")


if __name__ == "__main__":
    main()
