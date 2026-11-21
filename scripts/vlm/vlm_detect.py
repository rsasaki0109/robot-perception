#!/usr/bin/env python3
"""2D object detection with a pretrained vision-language model, on CPU.

Runs Florence-2 (no training, no GPU) on the camera frames of simulated
perception tasks (Robot Native Engine example 149 with --camera) and writes
its detections as `camera_batch` records, so the tracker can use them in
camera-assisted mode. Classes are the tracker's: person 1, cyclist 2, car 3.

Two prompts per frame:

1. `<OD>` (open object detection) for people. It names cars and bicycles
   alike "land vehicle", so its vehicles are not used.
2. `<CAPTION_TO_PHRASE_GROUNDING>` with "a car. a bicycle." for vehicles; it
   tells the two apart but returns fewer people than `<OD>`.

Grounding always returns a box for every phrase, even with nothing there, so
a grounded vehicle is kept only where `<OD>` also saw a vehicle (a "land
vehicle", a wheel, ...: IoU >= 0.3, or the `<OD>` box's centre inside it).
A bicycle box mostly inside a car box is dropped (the car's wheels confirm
it), a person box standing on a bicycle box becomes one cyclist box (their
union), and boxes of one class overlapping at IoU > 0.6 are merged.

The raw outputs of both prompts are saved (vlm_raw.jsonl), and
`--from-raw` redoes the post-processing from them without the model.

For each task directory it writes

    camera_detections.jsonl   one camera_batch per frame
    inputs_camera.jsonl       inputs.jsonl and the camera batches, by event time
    vlm_timing.json           seconds per frame

usage: vlm_detect.py <task-dir> [<task-dir> ...] [--model florence-community/Florence-2-base] [--threads 4] [--dtype bfloat16]

bfloat16 runs on CPUs with AMX or AVX512-BF16 (Sapphire Rapids and later)
several times faster than float32.

Needs torch (CPU build), transformers >= 4.56 and pillow.
"""

import argparse
import json
import time
from pathlib import Path

# <OD> labels kept as people.
PERSON_LABELS = {"person", "man", "woman", "boy", "girl", "human", "pedestrian"}
# Grounding phrases -> tracker classes.
VEHICLE_PROMPT = "a car. a bicycle."
VEHICLE_LABELS = {"a car": 3, "car": 3, "a bicycle": 2, "bicycle": 2}
# <OD> labels that confirm a grounded vehicle.
OD_VEHICLE_LABELS = {"land vehicle", "vehicle", "car", "van", "truck", "bus", "bicycle", "bicycle wheel",
                     "wheel", "tire", "motorcycle", "segway", "unicycle", "auto part"}
CAMERA_SENSOR_ID = 2
# Florence-2 reports no confidence; every kept box gets this score.
SCORE = 0.9


def load(model_id, threads, dtype):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(threads)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=getattr(torch, dtype)).eval()
    return processor, model


def prompt(processor, model, image, task, text=""):
    import torch

    inputs = processor(text=task + text, images=image, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=256, num_beams=1, do_sample=False)
    out = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(out, task=task, image_size=image.size)[task]
    return [(label.strip().lower(), [float(v) for v in box]) for label, box in zip(parsed["labels"], parsed["bboxes"])]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def run_prompts(processor, model, image):
    """Raw outputs of both prompts: `{"od": [[label, box]], "grounding": [[label, box]]}`."""
    return {
        "od": prompt(processor, model, image, "<OD>"),
        "grounding": prompt(processor, model, image, "<CAPTION_TO_PHRASE_GROUNDING>", VEHICLE_PROMPT),
    }


def inside(inner, outer):
    """Share of `inner`'s area inside `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


def corroborated(box, od_vehicles):
    for other in od_vehicles:
        cx, cy = (other[0] + other[2]) / 2, (other[1] + other[3]) / 2
        if iou(box, other) >= 0.3 or (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]):
            return True
    return False


def postprocess(raw):
    """`[(class, box, label)]` from the raw prompt outputs of one image."""
    people = [(1, box, label) for label, box in raw["od"] if label in PERSON_LABELS]
    od_vehicles = [box for label, box in raw["od"] if label in OD_VEHICLE_LABELS]
    vehicles = [(VEHICLE_LABELS[label], box, label) for label, box in raw["grounding"]
                if label in VEHICLE_LABELS and corroborated(box, od_vehicles)]
    # A car's own wheels confirm grounded "bicycles" on it: drop bicycles
    # that lie mostly inside a car.
    cars = [box for cls, box, _ in vehicles if cls == 3]
    vehicles = [v for v in vehicles if v[0] != 2 or all(inside(v[1], car) < 0.6 for car in cars)]
    # A rider: a person box over most of a bicycle's width, reaching down onto it.
    riders = set()
    merged = []
    for cls, bike, label in vehicles:
        if cls == 2:
            for k, (_, person, _) in enumerate(people):
                overlap = max(0.0, min(bike[2], person[2]) - max(bike[0], person[0]))
                if k not in riders and overlap > 0.5 * min(bike[2] - bike[0], person[2] - person[0]) \
                        and person[3] > bike[1] and person[1] < bike[1]:
                    riders.add(k)
                    bike = [min(bike[0], person[0]), min(bike[1], person[1]),
                            max(bike[2], person[2]), max(bike[3], person[3])]
                    label = "cyclist"
                    break
        merged.append((cls, bike, label))
    merged += [p for k, p in enumerate(people) if k not in riders]
    # One box per object: drop a box overlapping a kept one of its class.
    kept = []
    for cls, box, label in sorted(merged, key=lambda d: -(d[1][2] - d[1][0]) * (d[1][3] - d[1][1])):
        if all(c != cls or iou(box, b) <= 0.6 for c, b, _ in kept):
            kept.append((cls, box, label))
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dirs", nargs="+", type=Path)
    ap.add_argument("--model", default="florence-community/Florence-2-base")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--from-raw", action="store_true", help="reuse vlm_raw.jsonl instead of running the model")
    a = ap.parse_args()

    from PIL import Image

    processor, model = (None, None) if a.from_raw else load(a.model, a.threads, a.dtype)
    for task_dir in a.task_dirs:
        frames = [json.loads(line) for line in open(task_dir / "camera_frames.jsonl")]
        cached = {}
        if a.from_raw:
            cached = {r["timestamp_ns"]: r for r in map(json.loads, open(task_dir / "vlm_raw.jsonl"))}
        batches, seconds, raws = [], [], []
        for sequence, frame in enumerate(frames):
            if a.from_raw:
                raw = cached[frame["timestamp_ns"]]
            else:
                image = Image.open(task_dir / frame["image"]).convert("RGB")
                start = time.perf_counter()
                raw = {"timestamp_ns": frame["timestamp_ns"], **run_prompts(processor, model, image)}
                seconds.append(time.perf_counter() - start)
            raws.append(raw)
            found = postprocess(raw)
            detections = []
            for cls, box, label in found:
                x0, y0, x1, y1 = box
                if x1 - x0 < 1 or y1 - y0 < 1:
                    continue
                detections.append({"detection_id": len(detections), "bbox": [x0, y0, x1, y1],
                                   "class": cls, "score": SCORE, "label": label})
            t = frame["timestamp_ns"]
            batches.append({"type": "camera_batch", "schema_version": 1, "sensor_id": CAMERA_SENSOR_ID,
                            "sequence": sequence, "event_time_ns": t, "arrival_time_ns": t,
                            "frame_id": "camera", "detections": detections})
        if not a.from_raw:
            (task_dir / "vlm_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raws))
        with open(task_dir / "camera_detections.jsonl", "w") as f:
            for b in batches:
                f.write(json.dumps(b) + "\n")
        lidar = [json.loads(line) for line in open(task_dir / "inputs.jsonl")]
        merged = sorted(lidar + [{**b, "detections": [{k: v for k, v in d.items() if k != "label"}
                                                      for d in b["detections"]]} for b in batches],
                        key=lambda r: (r["event_time_ns"], r["sensor_id"]))
        with open(task_dir / "inputs_camera.jsonl", "w") as f:
            for r in merged:
                f.write(json.dumps(r) + "\n")
        if a.from_raw:
            print(f"{task_dir.name}: {len(frames)} frames post-processed from vlm_raw.jsonl")
            continue
        timing = {"model": a.model, "dtype": a.dtype, "frames": len(frames), "threads": a.threads,
                  "mean_s": sum(seconds) / max(1, len(seconds)), "total_s": sum(seconds)}
        (task_dir / "vlm_timing.json").write_text(json.dumps(timing, indent=2))
        print(f"{task_dir.name}: {len(frames)} frames, {timing['mean_s']:.2f} s/frame")


if __name__ == "__main__":
    main()
