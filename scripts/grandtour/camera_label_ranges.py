"""Attach independent RealSense range estimates to ETH-2 camera labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import zarr

from grandtour import Odometry, base_to_sensor
from pair_lidars import common_mask
from score_camera_labels import distortion_for, project_pixels


def reference(box, pixels, xyz, sensor_from_base):
    lower, upper = np.asarray(box).reshape(2, 2)
    center = (lower + upper) / 2
    inside = np.all(np.abs(pixels - center) <= (upper - lower) * 0.20, axis=1)
    samples = xyz[inside]
    if len(samples) < 25:
        return {"band": "unknown", "samples": len(samples)}
    ranges_m = np.linalg.norm(samples[:, :2], axis=1)
    q25, median, q75 = np.quantile(ranges_m, [0.25, 0.50, 0.75])
    if q75 - q25 > 0.6:
        return {"band": "unknown", "samples": len(samples), "iqr_m": round(q75 - q25, 3)}
    torso = np.median(samples, axis=0).reshape(1, 3)
    if not common_mask(torso, sensor_from_base)[0]:
        band = "outside_common_crop"
    elif median < 4:
        band = "0.5–4 m"
    elif median < 6:
        band = "4–6 m"
    else:
        band = "6–8 m"
    return {"band": band, "samples": len(samples), "range_m": round(median, 3),
            "iqr_m": round(q75 - q25, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("data", "depth-images", "labels", "pairs", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text())
    pairs = {row["mid360_scan"]: row["xt32_scan"]
             for row in map(json.loads, args.pairs.read_text().splitlines())}
    root = zarr.open_group(str(args.data), mode="r")
    tf = root["tf"].attrs["tf"]
    odom = Odometry(root["anymal_state_odometry"])
    camera = root[labels["camera"]]
    depth_camera = labels.get("depth_camera", "depth_camera_front_upper")
    depth = root[depth_camera]
    info = camera.attrs["camera_info"]
    k = np.array(info["K"]).reshape(3, 3)
    depth_k = np.array(depth.attrs["camera_info"]["K"]).reshape(3, 3)
    depth_times = depth["timestamp"][:]
    camera_from_base = base_to_sensor(tf, labels["camera"])
    base_from_depth = np.linalg.inv(base_to_sensor(tf, depth_camera))
    mid_from_base = base_to_sensor(tf, "livox_lidar")
    xt_from_base = base_to_sensor(tf, "hesai_lidar")
    scale = np.asarray(labels["image_size_px"]) / [info["width"], info["height"]]
    for index, frame in labels["frames"].items():
        i = int(index)
        camera_time = float(camera["timestamp"][frame.get("camera_frame", i)])
        depth_index = int(np.argmin(np.abs(depth_times - camera_time)))
        offset_ms = (depth_times[depth_index] - camera_time) * 1000
        if abs(offset_ms) > 35:
            raise ValueError(f"depth frame too far from camera frame {i}")
        with Image.open(args.depth_images / f"{depth_index:06d}.png") as image:
            depth_mm = np.asarray(image)
        if depth_mm.shape != (480, 848) or depth_mm.dtype != np.uint16:
            raise ValueError(f"expected a recorded uint16 D435i frame at {depth_index}")
        y, x = np.nonzero(depth_mm)
        z_m = depth_mm[y, x] * 0.001
        camera_rays = np.column_stack((x, y, np.ones_like(x)))
        xyz_depth = (camera_rays @ np.linalg.inv(depth_k).T) * z_m[:, None]
        world_from_depth = odom.at(depth_times[depth_index]) @ base_from_depth
        optical_from_depth = camera_from_base @ np.linalg.solve(
            odom.at(camera_time), world_from_depth
        )
        pixels, front = project_pixels(xyz_depth, optical_from_depth, k, distortion_for(info), scale)
        mid_pose = odom.at(root["livox_points_undistorted"]["timestamp"][i])
        mid_from_depth = np.linalg.solve(mid_pose, world_from_depth)
        xyz_mid = xyz_depth[front] @ mid_from_depth[:3, :3].T + mid_from_depth[:3, 3]
        xt_pose = odom.at(root["hesai_points_undistorted"]["timestamp"][pairs[i]])
        sensor_from_base = mid_from_base, xt_from_base @ np.linalg.solve(xt_pose, mid_pose)
        frame["depth_frame"] = depth_index
        frame["depth_minus_camera_ms"] = round(offset_ms, 2)
        frame["range_reference"] = [
            reference(box, pixels, xyz_mid, sensor_from_base) for box in frame["persons"]
        ]
    args.out.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
