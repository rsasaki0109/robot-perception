#!/usr/bin/env python3
"""Track and score every simulated perception task.

The tasks come from Robot Native Engine's example 149_perception_tasks, which
simulates scenes (people, cyclists, cars, structures), scans them with a
16-channel LiDAR, segments the scans with a cluster detector and writes one
directory per task in this repository's replay format:

    cargo run --release -p perception_tasks --example 149_perception_tasks -- --out <sim-dir>

This script replays each task through `robot-perception replay` in a few
configurations and scores the confirmed tracks against the simulated truth
with mot_metrics.py (HOTA, IDF1, MOTA, ID switches) and velocity_error.py.

Tasks simulated with a camera (`--camera`) and run through
scripts/vlm/vlm_detect.py have `inputs_camera.jsonl`; they are also replayed
in camera-assisted mode, where the VLM's 2D detections add class evidence to
the LiDAR tracks. Class accuracy is scored for every run: the LiDAR class of
each matched track, and the fused class (the camera's class prior when the
track has one, else the LiDAR class).

usage: evaluate_tasks.py <sim-dir> [--out <work-dir>] [--binary target/release/robot-perception] [--json results.json]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval"

# The ROS 2 node's defaults for people: distance association, no cascade.
BASE = ["--max-association-distance-m", "1.0", "--recent-track-window-ns", "0"]
VARIANTS = {
    "tracker": BASE,
    "+ 0.5 s smoothing": BASE + ["--smooth-lag", "5"],
    "+ pose correction": BASE + ["--pose-correction"],
}
# Variants that only make sense where the robot moves among landmarks.
ONLY_FOR = {"+ pose correction": {"moving_robot", "pose_jitter"}}
# Cars are seen from one side, so their centres are off by up to a metre.
MATCH_DISTANCE_M = {"mixed_traffic": 2.0}


def class_accuracy(output, truth_path, dmax):
    """Fraction of matched confirmed tracks whose class is right: (LiDAR, fused)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    truth = {}
    for line in open(truth_path):
        r = json.loads(line)
        if r.get("type") == "truth3d":
            truth[r["timestamp_ns"]] = r["objects"]
    lidar_ok = fused_ok = total = 0
    for line in open(output):
        f = json.loads(line)
        if f.get("type") != "lidar_frame" or f["timestamp_ns"] not in truth:
            continue
        objects = truth[f["timestamp_ns"]]
        tracks = [t for t in f["tracks"] if t["lifecycle"] == "confirmed"]
        if not objects or not tracks:
            continue
        d = np.linalg.norm(np.array([o["position"][:3] for o in objects])[:, None]
                           - np.array([t["position"][:3] for t in tracks])[None], axis=2)
        cost = np.where(d <= dmax, d, 1e6)
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] >= 1e6:
                continue
            track, want = tracks[j], objects[i]["class"]
            prior = track.get("class_prior")
            fused = int(prior[0]) if prior else track.get("class_label")
            total += 1
            lidar_ok += track.get("class_label") == want
            fused_ok += fused == want
    return (lidar_ok / total, fused_ok / total) if total else (None, None)


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{result.stderr}")
    return result.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sim_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("sim-out"))
    ap.add_argument("--binary", default="target/release/robot-perception")
    ap.add_argument("--json", type=Path, help="also write all results here")
    a = ap.parse_args()

    manifest = json.loads((a.sim_dir / "tasks.json").read_text())
    rows = []
    for task in manifest:
        name = task["task"]
        src = a.sim_dir / name
        dmax = str(MATCH_DISTANCE_M.get(name, 1.0))
        variants = list(VARIANTS.items())
        if (src / "inputs_camera.jsonl").exists():
            variants.append(("+ VLM camera", BASE))
        for variant, flags in variants:
            if variant in ONLY_FOR and name not in ONLY_FOR[variant]:
                continue
            camera = variant == "+ VLM camera"
            inputs = src / ("inputs_camera.jsonl" if camera else "inputs.jsonl")
            mode = "camera_assisted" if camera else "lidar_only"
            work = a.out / name / variant.strip("+ ").replace(" ", "_").replace(".", "")
            work.mkdir(parents=True, exist_ok=True)
            output = work / "output.jsonl"
            run([a.binary, "replay", "--setup", src / "setup.json", "--inputs", inputs,
                 "--mode", mode, "--output", output, "--diagnostics", work / "diagnostics.json", *flags])
            mot = json.loads(run([sys.executable, EVAL / "mot_metrics.py", output, src / "truth.jsonl",
                                  "--max-distance", dmax, "--json"]))
            vel = json.loads(run([sys.executable, EVAL / "velocity_error.py", output, src / "truth.jsonl",
                                  "--max-distance", dmax, "--json"]))
            lidar_class, fused_class = class_accuracy(output, src / "truth.jsonl", float(dmax))
            rows.append({"task": name, "variant": variant, "detection_recall": task["detection_recall"],
                         "detection_precision": task["detection_precision"], **mot,
                         "velocity_median": vel["median"], "velocity_p90": vel["p90"],
                         "class_accuracy_lidar": lidar_class, "class_accuracy_fused": fused_class})

    f = lambda v: "-" if v is None else f"{v:.3f}"
    print("| task | tracker | detection recall | detection precision | HOTA | IDF1 | MOTA | ID switches "
          "| velocity median (m/s) | class accuracy |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        cls = r["class_accuracy_fused"] if r["variant"] == "+ VLM camera" else r["class_accuracy_lidar"]
        print(f"| {r['task']} | {r['variant']} | {r['detection_recall']:.3f} | {r['detection_precision']:.3f} "
              f"| {r['HOTA']:.3f} | {r['IDF1']:.3f} | {r['MOTA']:.3f} | {int(r['IDSW'])} | {r['velocity_median']:.3f} "
              f"| {f(cls)} |")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
