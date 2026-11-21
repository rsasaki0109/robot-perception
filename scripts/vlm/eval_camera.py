#!/usr/bin/env python3
"""Score 2D camera detections against the renderer's exact boxes.

Ground truth comes from `camera_frames.jsonl` (the item-id map of the CPU
ray tracer: the visible part of every object, at least 30 pixels).
Detections come from `camera_detections.jsonl` (vlm_detect.py). A detection
matches a truth box at IoU >= 0.5 (greedy, highest IoU first, one to one,
any class); class accuracy is over the matched pairs. Recall is also given by
the truth box height in pixels.

usage: eval_camera.py <task-dir> [<task-dir> ...] [--json out.json]
"""

import argparse
import json
from pathlib import Path

SIZES = [("< 40 px", 0, 40), ("40-100 px", 40, 100), (">= 100 px", 100, 10**9)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score(task_dir):
    truth = {json.loads(l)["timestamp_ns"]: json.loads(l) for l in open(task_dir / "camera_frames.jsonl")}
    found = {json.loads(l)["event_time_ns"]: json.loads(l) for l in open(task_dir / "camera_detections.jsonl")}
    n_truth = n_det = tp = class_ok = 0
    by_size = {name: [0, 0] for name, _, _ in SIZES}
    by_class = {}
    for t, frame in truth.items():
        objects = frame["objects"]
        detections = found.get(t, {"detections": []})["detections"]
        n_truth += len(objects)
        n_det += len(detections)
        pairs = sorted(((iou(o["bbox"], d["bbox"]), i, j) for i, o in enumerate(objects)
                        for j, d in enumerate(detections)), reverse=True)
        used_o, used_d = set(), set()
        for value, i, j in pairs:
            if value < 0.5:
                break
            if i in used_o or j in used_d:
                continue
            used_o.add(i)
            used_d.add(j)
            tp += 1
            class_ok += objects[i]["class"] == detections[j]["class"]
        for i, o in enumerate(objects):
            h = o["bbox"][3] - o["bbox"][1]
            for name, lo, hi in SIZES:
                if lo <= h < hi:
                    by_size[name][0] += i in used_o
                    by_size[name][1] += 1
            c = by_class.setdefault(o["class"], [0, 0])
            c[0] += i in used_o
            c[1] += 1
    ratio = lambda a, b: a / b if b else None
    return {
        "task": task_dir.name, "frames": len(truth), "truth_boxes": n_truth, "detections": n_det,
        "recall": ratio(tp, n_truth), "precision": ratio(tp, n_det), "class_accuracy": ratio(class_ok, tp),
        "recall_by_height": {k: ratio(*v) for k, v in by_size.items()},
        "recall_by_class": {str(k): ratio(*v) for k, v in sorted(by_class.items())},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rows = [score(d) for d in a.task_dirs if (d / "camera_detections.jsonl").exists()]
    f = lambda v: "-" if v is None else f"{v:.3f}"
    print("| task | truth boxes | recall | precision | class accuracy | recall < 40 px | 40-100 px | >= 100 px |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        s = r["recall_by_height"]
        print(f"| {r['task']} | {r['truth_boxes']} | {f(r['recall'])} | {f(r['precision'])} | {f(r['class_accuracy'])} "
              f"| {f(s['< 40 px'])} | {f(s['40-100 px'])} | {f(s['>= 100 px'])} |")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
