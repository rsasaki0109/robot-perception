#!/usr/bin/env python3
"""Make detector-blind ETH-2 review sheets from paired scans and raw camera images."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from detect import mid360_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--three", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / name).exists() for name in ("frame_review.csv", "people.csv")):
        parser.error("choose a new output directory to preserve existing review labels")

    # Uniform selection by scan ID, independent of any detector's output.
    frames = range(3700, 3901, 20)
    for frame in frames:
        stem = f"{frame:06d}"
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained")
        with Image.open(args.images / f"{stem}.jpeg") as image:
            axes[0].imshow(image)
        axes[0].set_title("Front image (~48 ms after scan)")
        axes[0].axis("off")
        for axis, scans, title in zip(axes[1:], (args.single, args.three),
                                      ("Single sweep", "Three sweeps")):
            points = mid360_points(scans / f"{stem}.bin", 1.0)
            x, y, z = points[:, :3].T
            keep = (np.hypot(x, y) <= 8) & (z >= -0.35) & (z <= 1.4)
            axis.scatter(y[keep], x[keep], c=z[keep], cmap="viridis",
                         s=1, vmin=-0.35, vmax=1.4, linewidths=0)
            axis.plot(0, 0, "^", color="black")
            axis.set(xlim=(8, -8), ylim=(-8, 8), aspect="equal",
                     xlabel="Left (m)", ylabel="Forward (m)", title=title)
            axis.grid(alpha=0.2)
        fig.suptitle(f"GrandTour ETH-2 • scan {frame} • color by height (m)")
        fig.savefig(args.out / f"{stem}.png", dpi=145)
        plt.close(fig)

    with (args.out / "frame_review.csv").open("x", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["scan", "reviewed_scope", "status", "notes"])
        writer.writerows((frame, "", "pending", "") for frame in frames)
    with (args.out / "people.csv").open("x", newline="") as output:
        csv.writer(output).writerow(
            ["scan", "person_id", "forward_m", "left_m", "motion", "occlusion", "notes"]
        )


if __name__ == "__main__":
    main()
