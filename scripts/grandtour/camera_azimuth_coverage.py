"""Measure nominal angular coverage for the ETH-2 camera review regions."""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from grandtour import base_to_sensor
from score_camera_labels import distortion_for, project_pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    root = zarr.open_group(str(args.data), mode="r")
    tf = root["tf"].attrs["tf"]
    azimuth_deg = np.arange(-180, 180)
    angle = np.deg2rad(azimuth_deg)
    cameras = {}
    for name in ("alphasense_front_center", "alphasense_left", "alphasense_right",
                 "depth_camera_rear_upper"):
        info = root[name].attrs["camera_info"]
        cameras[name] = (
            base_to_sensor(tf, name), np.array(info["K"]).reshape(3, 3),
            distortion_for(info), np.array([480, 360]) / [info["width"], info["height"]],
        )

    for range_m in (2, 4, 6, 8):
        for z_m in (0.0, 0.4, 0.8):
            points = np.column_stack((
                range_m * np.cos(angle), range_m * np.sin(angle),
                np.full(len(angle), z_m),
            ))
            rgb = np.zeros(len(angle), dtype=bool)
            for name, calibration in cameras.items():
                uv, front = project_pixels(points, *calibration)
                rear = name == "depth_camera_rear_upper"
                lower, upper = ([0, 0], [480, 360]) if rear else ([20, 0], [460, 295])
                keep = front.copy()
                keep[front] = ((uv >= lower) & (uv < upper)).all(axis=1)
                if rear:
                    print(json.dumps(dict(
                        range_m=range_m, z_m=z_m, rgb_bins=int(rgb.sum()),
                        rear_depth_bins=int(keep.sum()), union_bins=int((rgb | keep).sum()),
                    )))
                else:
                    rgb |= keep


if __name__ == "__main__":
    main()
