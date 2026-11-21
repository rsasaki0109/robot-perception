#!/usr/bin/env python3
"""Track and score every simulated perception task.

The tasks come from Robot Native Engine's example 149_perception_tasks, which
simulates scenes (people, cyclists, cars, structures), scans them with a
16-channel LiDAR, segments the scans with a cluster detector and writes one
directory per task in this repository's replay format:

    cargo run --release -p perception_tasks --example 149_perception_tasks -- --out <sim-dir>

This script replays each task through `robo-perception replay` in a few
configurations and scores the confirmed tracks against the simulated truth
with mot_metrics.py (HOTA, IDF1, MOTA, ID switches) and velocity_error.py.

usage: evaluate_tasks.py <sim-dir> [--out <work-dir>] [--binary target/release/robo-perception] [--json results.json]
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


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{result.stderr}")
    return result.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sim_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("sim-out"))
    ap.add_argument("--binary", default="target/release/robo-perception")
    ap.add_argument("--json", type=Path, help="also write all results here")
    a = ap.parse_args()

    manifest = json.loads((a.sim_dir / "tasks.json").read_text())
    rows = []
    for task in manifest:
        name = task["task"]
        src = a.sim_dir / name
        dmax = str(MATCH_DISTANCE_M.get(name, 1.0))
        for variant, flags in VARIANTS.items():
            if variant in ONLY_FOR and name not in ONLY_FOR[variant]:
                continue
            work = a.out / name / variant.strip("+ ").replace(" ", "_").replace(".", "")
            work.mkdir(parents=True, exist_ok=True)
            output = work / "output.jsonl"
            run([a.binary, "replay", "--setup", src / "setup.json", "--inputs", src / "inputs.jsonl",
                 "--mode", "lidar_only", "--output", output, "--diagnostics", work / "diagnostics.json", *flags])
            mot = json.loads(run([sys.executable, EVAL / "mot_metrics.py", output, src / "truth.jsonl",
                                  "--max-distance", dmax, "--json"]))
            vel = json.loads(run([sys.executable, EVAL / "velocity_error.py", output, src / "truth.jsonl",
                                  "--max-distance", dmax, "--json"]))
            rows.append({"task": name, "variant": variant, "detection_recall": task["detection_recall"],
                         "detection_precision": task["detection_precision"], **mot,
                         "velocity_median": vel["median"], "velocity_p90": vel["p90"]})

    print("| task | tracker | detection recall | detection precision | HOTA | IDF1 | MOTA | ID switches | velocity median (m/s) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['task']} | {r['variant']} | {r['detection_recall']:.3f} | {r['detection_precision']:.3f} "
              f"| {r['HOTA']:.3f} | {r['IDF1']:.3f} | {r['MOTA']:.3f} | {int(r['IDSW'])} | {r['velocity_median']:.3f} |")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
