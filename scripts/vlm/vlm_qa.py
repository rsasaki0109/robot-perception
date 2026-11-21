#!/usr/bin/env python3
"""Scene questions to a pretrained vision-language model, scored against simulator truth.

Asks a small chat VLM (default SmolVLM-256M-Instruct, CPU, no training) two
questions about each camera frame of a simulated perception task (Robot
Native Engine example 149 with --camera):

- count: "How many people are in this image?"; truth is the number of people
  with at least --min-pixels visible pixels in the renderer's item-id map.
- path: "Is a person in the robot's path within 5 m?"; truth is whether a
  person stands less than 5 m ahead of the robot and within 1 m of its
  heading line, from the 3D truth and the ego pose of the frame.

When the task also has camera_detections.jsonl (vlm_detect.py), the count is
also taken from the number of person boxes Florence-2 found, for comparison.

usage: vlm_qa.py <task-dir> [<task-dir> ...] [--model HuggingFaceTB/SmolVLM-256M-Instruct]
                 [--every 2] [--dtype bfloat16] [--json out.json]
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

COUNT_Q = "How many people are in this image? Answer with a single number."
PATH_Q = ("The camera looks straight ahead along the robot's path. Is there a person standing in the "
          "path less than 5 meters ahead? Answer yes or no.")
PATH_AHEAD_M = 5.0
PATH_HALF_WIDTH_M = 1.0
WORDS = {"zero": 0, "no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def load(model_id, dtype, threads, split_images):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(threads)
    # One 512 px view of the frame; splitting it into tiles costs ten times
    # as long per question.
    processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=split_images)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=getattr(torch, dtype)).eval()
    return processor, model


def ask(processor, model, image, question):
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: (v.to(model.dtype) if v.is_floating_point() else v) for k, v in inputs.items()}
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=12, do_sample=False)
    answer = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return answer.strip()


def parse_count(answer):
    m = re.search(r"\d+", answer)
    if m:
        return int(m.group())
    for word in re.findall(r"[a-z]+", answer.lower()):
        if word in WORDS:
            return WORDS[word]
    return None


def parse_yes(answer):
    a = answer.strip().lower()
    if a.startswith("yes"):
        return True
    if a.startswith("no"):
        return False
    return None


def path_truth(task_dir):
    """Per timestamp: is a person within the path box ahead of the robot?"""
    setup = json.load(open(task_dir / "setup.json"))
    poses = {p["timestamp_ns"]: np.array(p["transform"]).reshape(4, 4) for p in setup["poses"]}
    blocked = {}
    for line in open(task_dir / "truth.jsonl"):
        r = json.loads(line)
        pose = poses[r["timestamp_ns"]]
        inv = np.linalg.inv(pose)
        hit = False
        for o in r["objects"]:
            if o["class"] != 1:
                continue
            p = inv @ np.array([*o["position"][:3], 1.0])
            hit |= 0.0 < p[0] < PATH_AHEAD_M and abs(p[1]) < PATH_HALF_WIDTH_M
        blocked[r["timestamp_ns"]] = bool(hit)
    return blocked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dirs", nargs="+", type=Path)
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--every", type=int, default=2, help="ask about every N-th camera frame")
    ap.add_argument("--min-pixels", type=int, default=300)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--split-images", action="store_true", help="let the processor tile each frame")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    from PIL import Image

    processor, model = load(a.model, a.dtype, a.threads, a.split_images)
    rows = []
    for task_dir in a.task_dirs:
        frames = [json.loads(l) for l in open(task_dir / "camera_frames.jsonl")][:: a.every]
        florence = {}
        if (task_dir / "camera_detections.jsonl").exists():
            for l in open(task_dir / "camera_detections.jsonl"):
                b = json.loads(l)
                florence[b["event_time_ns"]] = sum(d["class"] == 1 for d in b["detections"])
        blocked = path_truth(task_dir)
        records, seconds = [], []
        for frame in frames:
            image = Image.open(task_dir / frame["image"]).convert("RGB")
            start = time.perf_counter()
            count_answer = ask(processor, model, image, COUNT_Q)
            path_answer = ask(processor, model, image, PATH_Q)
            seconds.append(time.perf_counter() - start)
            records.append({
                "timestamp_ns": frame["timestamp_ns"],
                "count_truth": sum(o["class"] == 1 and o["visible_pixels"] >= a.min_pixels for o in frame["objects"]),
                "count_vlm": parse_count(count_answer), "count_answer": count_answer,
                "count_detector": florence.get(frame["timestamp_ns"]),
                "path_truth": blocked[frame["timestamp_ns"]],
                "path_vlm": parse_yes(path_answer), "path_answer": path_answer,
            })
        (task_dir / "vlm_qa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

        def count_stats(key):
            pairs = [(r[key], r["count_truth"]) for r in records if r[key] is not None]
            if not pairs:
                return None, None
            exact = sum(p == t for p, t in pairs) / len(records)
            mae = sum(abs(p - t) for p, t in pairs) / len(pairs)
            return exact, mae

        vlm_exact, vlm_mae = count_stats("count_vlm")
        det_exact, det_mae = count_stats("count_detector")
        answered = [r for r in records if r["path_vlm"] is not None]
        positives = [r for r in records if r["path_truth"]]
        rows.append({
            "task": task_dir.name, "frames": len(records), "model": a.model,
            "count_exact_vlm": vlm_exact, "count_mae_vlm": vlm_mae,
            "count_exact_detector": det_exact, "count_mae_detector": det_mae,
            "path_accuracy": sum(r["path_vlm"] == r["path_truth"] for r in answered) / len(records) if records else None,
            "path_positive_frames": len(positives),
            "path_recall": (sum(r["path_vlm"] is True for r in positives) / len(positives)) if positives else None,
            "seconds_per_frame": sum(seconds) / max(1, len(seconds)),
        })
        print(f"{task_dir.name}: {len(records)} frames, {rows[-1]['seconds_per_frame']:.2f} s/frame", flush=True)

    f = lambda v: "-" if v is None else f"{v:.2f}"
    print("| task | frames | count exact (VLM) | count MAE (VLM) | count exact (Florence boxes) | count MAE (Florence boxes) "
          "| path accuracy | path frames with a person | path recall |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['task']} | {r['frames']} | {f(r['count_exact_vlm'])} | {f(r['count_mae_vlm'])} "
              f"| {f(r['count_exact_detector'])} | {f(r['count_mae_detector'])} | {f(r['path_accuracy'])} "
              f"| {r['path_positive_frames']} | {f(r['path_recall'])} |")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
