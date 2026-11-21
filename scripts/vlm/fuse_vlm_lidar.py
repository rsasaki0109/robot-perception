#!/usr/bin/env python3
"""Merge LiDAR detections that a VLM sees as one car or cyclist (before tracking).

A geometric LiDAR detector splits a car partly hidden by a passer-by into
person-sized pieces, and a cyclist seen end-on looks like a person. The camera
cannot fix that inside the tracker: it only adds class evidence to tracks the
LiDAR already made (docs/vlm.md). This step fixes it in the detections.

For each LiDAR scan that has a camera frame at the same time:

1. Each LiDAR detection's 3D box (position + `shape.corners`) is projected
   into the image with the calibration in setup.json.
2. It is assigned to the VLM box (camera_detections.jsonl) that covers the
   largest share of its projected box, if that share is >= --min-cover. A
   person box wins a tie with a vehicle box: a pedestrian in front of a car
   stays a pedestrian.
3. The detections assigned to one car or cyclist box are merged into one
   detection with the VLM class: its footprint is the bounding box of all
   their corners along their principal axis, its height the tallest. The
   LiDAR sees only the near face, so the box is then completed to the class's
   typical size (car 4.4 x 1.8 m, bicycle 1.7 x 0.5 m): if it is shallower
   along the line of sight than the object would be (end-on: its length;
   side-on: its width), its centre moves away from the sensor by half the
   missing depth. Pieces
   farther than --max-spread-m from the nearest piece are left alone (an object
   behind the car in the same image region).
4. A single detection assigned to a person box takes the person class.

Scans without a camera frame pass through unchanged. Writes
`inputs_fused.jsonl` (LiDAR batches only) and, if camera batches exist,
`inputs_fused_camera.jsonl` (with them).

usage: fuse_vlm_lidar.py <task-dir> [<task-dir> ...] [--min-cover 0.5]
"""

import argparse
import json
from pathlib import Path

import numpy as np

SPREAD_M = {2: 2.5, 3: 6.0}
# Typical (length, width) by class, for completing a box seen from one side.
SIZE_M = {2: (1.7, 0.5), 3: (4.4, 1.8)}


def calibration(setup):
    m = np.array(setup["camera_to_reference"], float).reshape(4, 4)
    lidar_to_camera = np.linalg.inv(m)
    k = setup["intrinsics"]
    return lidar_to_camera, (k["fx"], k["fy"], k["cx"], k["cy"]), (k["image_width"], k["image_height"])


def project_box(points, lidar_to_camera, intrinsics):
    """Image box of LiDAR-frame points, or None if any point is behind the camera."""
    p = (lidar_to_camera[:3, :3] @ points.T).T + lidar_to_camera[:3, 3]
    if np.any(p[:, 2] <= 0.1):
        return None
    fx, fy, cx, cy = intrinsics
    u = fx * p[:, 0] / p[:, 2] + cx
    v = fy * p[:, 1] / p[:, 2] + cy
    return [u.min(), v.min(), u.max(), v.max()]


def cover(inner, outer):
    """Share of `inner`'s area inside `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


def corners_of(detection):
    c = np.array(detection["shape"]["corners"], float).reshape(8, 3)
    return c + np.array(detection["position"], float)


def merge(group, cls):
    """One detection from several, all in the LiDAR frame."""
    corners = np.vstack([corners_of(d) for d in group])
    xy = corners[:, :2]
    center = xy.mean(axis=0)
    cov = np.cov((xy - center).T) if len(xy) > 2 else np.eye(2)
    angle = 0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    axes = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    local = (xy - center) @ axes.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    mid = center + ((lo + hi) / 2) @ axes
    if cls in SIZE_M:
        # Depth of the seen box along the line of sight, and the object's.
        ray = mid / max(np.linalg.norm(mid), 1e-6)
        seen = np.ptp(xy @ ray)
        length, width = SIZE_M[cls]
        # A long visible face means a side view; a short one, an end-on view.
        expected = width if np.ptp(local[:, 0]) > 0.6 * length else length
        shift = max(0.0, expected - seen) / 2
        mid = mid + ray * shift
        xy = np.vstack([xy, xy + ray * 2 * shift])
        local = (xy - center) @ axes.T
        lo, hi = local.min(axis=0), local.max(axis=0)
    z_lo, z_hi = corners[:, 2].min(), corners[:, 2].max()
    position = [float(mid[0]), float(mid[1]), float(np.mean([d["position"][2] for d in group]))]
    new_corners = []
    for k in range(8):
        u = hi[0] if k & 1 else lo[0]
        v = hi[1] if k & 2 else lo[1]
        z = z_hi if k & 4 else z_lo
        p = mid + (np.array([u, v]) - (lo + hi) / 2) @ axes
        new_corners += [float(p[0] - position[0]), float(p[1] - position[1]), float(z - position[2])]
    rng = float(np.hypot(*position[:2]))
    sigma = 0.08 + 0.01 * rng + 0.25 * (len(group) > 1)
    size = hi - lo
    return {
        "position": position,
        "covariance": [sigma ** 2, 0, 0, 0, sigma ** 2, 0, 0, 0, 0.04],
        "class": cls,
        "score": max(d["score"] for d in group),
        "shape": {"height": float(max(z_hi - z_lo, 0.05)), "width": float(max(size[1], 0.05)),
                  "length": float(max(size[0], 0.05)), "rotation_y": 0.0, "corners": new_corners},
    }


def fuse_batch(batch, camera, lidar_to_camera, intrinsics, min_cover):
    detections = batch["detections"]
    boxes = camera["detections"]
    # Smaller boxes first, people before vehicles: the most specific box wins ties.
    order = sorted(range(len(boxes)), key=lambda j: (boxes[j]["class"] != 1,
                   (boxes[j]["bbox"][2] - boxes[j]["bbox"][0]) * (boxes[j]["bbox"][3] - boxes[j]["bbox"][1])))
    assigned = {}
    for i, d in enumerate(detections):
        if "shape" not in d or not d["shape"].get("corners"):
            continue
        projected = project_box(corners_of(d), lidar_to_camera, intrinsics)
        if projected is None:
            continue
        best, best_cover = None, min_cover
        for j in order:
            c = cover(projected, boxes[j]["bbox"])
            if c > best_cover + 1e-9:
                best, best_cover = j, c
        if best is not None:
            assigned.setdefault(best, []).append(i)
    out, used, stats = [], set(), {"merged_groups": 0, "merged_pieces": 0, "relabelled": 0}
    for j, members in assigned.items():
        cls = boxes[j]["class"]
        if cls == 1:
            if len(members) == 1:
                d = dict(detections[members[0]])
                if d.get("class") != 1:
                    stats["relabelled"] += 1
                d["class"] = 1
                out.append(d)
                used.add(members[0])
            continue
        # Keep the pieces near the nearest one (others may be behind the object).
        near = min(members, key=lambda i: np.hypot(*detections[i]["position"][:2]))
        anchor = np.array(detections[near]["position"][:2])
        group = [i for i in members
                 if np.linalg.norm(np.array(detections[i]["position"][:2]) - anchor) <= SPREAD_M[cls]]
        if len(group) > 1:
            stats["merged_groups"] += 1
            stats["merged_pieces"] += len(group)
        elif detections[group[0]].get("class") != cls:
            stats["relabelled"] += 1
        out.append(merge([detections[i] for i in group], cls))
        used.update(group)
    out += [d for i, d in enumerate(detections) if i not in used]
    return {**batch, "detections": out}, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dirs", nargs="+", type=Path)
    ap.add_argument("--min-cover", type=float, default=0.5)
    a = ap.parse_args()
    for task_dir in a.task_dirs:
        lidar_to_camera, intrinsics, _ = calibration(json.load(open(task_dir / "setup.json")))
        camera = {r["event_time_ns"]: r for r in map(json.loads, open(task_dir / "camera_detections.jsonl"))}
        fused, totals, frames = [], {}, 0
        for line in open(task_dir / "inputs.jsonl"):
            batch = json.loads(line)
            if batch.get("type") == "lidar_batch" and batch["event_time_ns"] in camera:
                frames += 1
                batch, stats = fuse_batch(batch, camera[batch["event_time_ns"]], lidar_to_camera, intrinsics,
                                          a.min_cover)
                for k, v in stats.items():
                    totals[k] = totals.get(k, 0) + v
            fused.append(batch)
        (task_dir / "inputs_fused.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fused))
        cams = [dict(c, detections=[{k: v for k, v in d.items() if k != "label"} for d in c["detections"]])
                for c in camera.values()]
        merged = sorted(fused + cams, key=lambda r: (r["event_time_ns"], r["sensor_id"]))
        (task_dir / "inputs_fused_camera.jsonl").write_text("".join(json.dumps(r) + "\n" for r in merged))
        print(f"{task_dir.name}: {frames} scans with a camera frame, {totals}")


if __name__ == "__main__":
    main()
