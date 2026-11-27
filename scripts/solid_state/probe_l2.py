#!/usr/bin/env python3
"""Inspect Unitree's L2 ROS 1 sample and optionally export the first scan as XYZI."""

import argparse
from itertools import islice
from pathlib import Path

import numpy as np


def packed_xyzi(message):
    names = ("x", "y", "z", "intensity")
    fields = {field.name: field for field in message.fields}
    if any(name not in fields or fields[name].datatype != 7 or fields[name].count != 1
           for name in names):
        raise ValueError("the cloud must contain scalar float32 x, y, z and intensity")
    dtype = np.dtype(dict(
        names=names,
        formats=[">f4" if message.is_bigendian else "<f4"] * len(names),
        offsets=[fields[name].offset for name in names],
        itemsize=message.point_step,
    ))
    cloud = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                       strides=(message.row_step, message.point_step))
    points = np.column_stack([cloud[name].ravel() for name in names])
    return points[np.isfinite(points).all(axis=1)].astype("<f4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--export-first", type=Path)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    scans, near, closest = [], [], []
    with Reader(args.bag) as bag:
        cloud = [c for c in bag.connections if c.topic == "/unilidar/cloud"
                 and c.msgtype == "sensor_msgs/msg/PointCloud2"]
        if len(cloud) != 1:
            parser.error("expected exactly one L2 /unilidar/cloud PointCloud2 stream")
        print(f"{cloud[0].topic}: {cloud[0].msgcount} recorded frames")
        for i, (connection, _, raw) in enumerate(islice(bag.messages(connections=cloud), args.frames)):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            points = packed_xyzi(message)
            ranges = np.linalg.norm(points[:, :3], axis=1)
            scans.append(len(ranges))
            near.append(int(np.count_nonzero(ranges < 0.5)))
            if len(ranges):
                closest.append(float(ranges.min()))
            if i == 0:
                print(f"frame={message.header.frame_id}, point_step={message.point_step} bytes")
                if args.export_first:
                    with args.export_first.open("xb") as target:
                        points.tofile(target)
    if not closest:
        parser.error("no finite returns in the inspected scans")
    print(f"{len(scans)} frames: {min(scans)}–{max(scans)} points/frame, "
          f"closest {min(closest):.3f} m, "
          f"returns below 0.5 m in {sum(count > 0 for count in near)} frames")


if __name__ == "__main__":
    main()
