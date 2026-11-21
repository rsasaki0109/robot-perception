#!/usr/bin/env python3
"""Recall and precision of LiDAR detections against the simulated truth.

A detection is a hit when it lies within 0.6 m (person), 1.0 m (cyclist) or
2.0 m (car) of a truth object of the same class, horizontally, matched one to
one (closest first). This is the detector score of Robot Native Engine example
149, so fused detections (fuse_vlm_lidar.py) can be compared with the raw ones.
Optionally only the scans that have a camera frame are scored, and only the
truth objects whose centre projects into the camera image (--in-view; a
detection is then counted only if it is in view too).

usage: eval_lidar_detections.py <task-dir> [--inputs inputs.jsonl inputs_fused.jsonl] [--camera-scans-only] [--in-view]
"""

import argparse
import json
from pathlib import Path

import numpy as np

MATCH_M = {1: 0.6, 2: 1.0, 3: 2.0}


def in_view_test(setup):
    """Whether a `map`-frame point projects into the image, given the scan pose."""
    lidar_to_camera = np.linalg.inv(np.array(setup["camera_to_reference"], float).reshape(4, 4))
    k = setup["intrinsics"]

    def test(pose, point):
        p_lidar = np.linalg.inv(pose) @ np.array([*point[:2], 0.9, 1.0])
        c = lidar_to_camera @ p_lidar
        if c[2] <= 0.5:
            return False
        u = k["fx"] * c[0] / c[2] + k["cx"]
        return 0 <= u < k["image_width"]

    return test


def score(task_dir, inputs_name, scans, in_view=False):
    setup = json.load(open(task_dir / "setup.json"))
    visible = in_view_test(setup) if in_view else None
    poses = {p["timestamp_ns"]: np.array(p["transform"]).reshape(4, 4) for p in setup["poses"]}
    truth = {r["timestamp_ns"]: r["objects"] for r in map(json.loads, open(task_dir / "truth.jsonl"))}
    n_truth = n_det = hits = 0
    per_class = {c: [0, 0] for c in MATCH_M}
    for line in open(task_dir / inputs_name):
        b = json.loads(line)
        if b.get("type") != "lidar_batch" or (scans is not None and b["event_time_ns"] not in scans):
            continue
        pose = poses[b["event_time_ns"]]
        dets = [((pose[:3, :3] @ np.array(d["position"]) + pose[:3, 3])[:2], d.get("class")) for d in b["detections"]]
        objs = [(np.array(o["position"][:2]), o["class"]) for o in truth[b["event_time_ns"]]]
        if visible is not None:
            dets = [d for d in dets if visible(pose, d[0])]
            objs = [o for o in objs if visible(pose, o[0])]
        n_truth += len(objs)
        n_det += len(dets)
        pairs = sorted((np.linalg.norm(p - q), i, j) for i, (p, c) in enumerate(dets) for j, (q, k) in enumerate(objs)
                       if c == k and np.linalg.norm(p - q) <= MATCH_M[k])
        ui, uj = set(), set()
        for _, i, j in pairs:
            if i not in ui and j not in uj:
                ui.add(i)
                uj.add(j)
                hits += 1
        for j, (_, k) in enumerate(objs):
            per_class[k][0] += j in uj
            per_class[k][1] += 1
    return {"inputs": inputs_name, "recall": hits / max(1, n_truth), "precision": hits / max(1, n_det),
            "recall_by_class": {str(k): (v[0] / v[1] if v[1] else None) for k, v in per_class.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--inputs", nargs="+", default=["inputs.jsonl", "inputs_fused.jsonl"])
    ap.add_argument("--camera-scans-only", action="store_true")
    ap.add_argument("--in-view", action="store_true", help="only objects and detections in the camera image")
    a = ap.parse_args()
    scans = None
    if a.camera_scans_only:
        scans = {json.loads(l)["event_time_ns"] for l in open(a.task_dir / "camera_detections.jsonl")}
    for name in a.inputs:
        r = score(a.task_dir, name, scans, a.in_view)
        cls = "  ".join(f"class {k}: {'-' if v is None else f'{v:.3f}'}" for k, v in r["recall_by_class"].items())
        print(f"{a.task_dir.name} {name}: recall {r['recall']:.3f}  precision {r['precision']:.3f}  ({cls})")


if __name__ == "__main__":
    main()
