# Pretrained vision-language models on CPU

Two uses of pretrained vision-language models (VLMs) on the camera of the
simulated perception tasks ([`docs/simulation.md`](simulation.md)). Neither
needs training or a GPU:

1. **2D detection** with Florence-2-base (0.23 B parameters), fed to the
   tracker's camera-assisted mode;
2. **scene questions** ("how many people?", "is the path clear?") to
   SmolVLM-256M-Instruct.

Ground truth comes from the simulator. The 2D boxes are the exact visible
extent of each object, from the renderer's item-id map; the path question is
scored against 3D positions and the ego pose.

## Reproduce

```bash
# RobotNativeEngine: simulate the tasks with a forward camera (about 2 min on 4 cores)
cargo run --release -p perception_tasks --example 149_perception_tasks -- --camera --out /tmp/tasks
# here, in a Python environment with torch (CPU build), transformers >= 4.56 and pillow
python scripts/vlm/vlm_detect.py /tmp/tasks/* --dtype bfloat16      # 2D detections + camera inputs
python scripts/vlm/eval_camera.py /tmp/tasks/*                       # 2D scores
python scripts/vlm/vlm_qa.py /tmp/tasks/<task> ...                    # scene questions
python3 scripts/sim/evaluate_tasks.py /tmp/tasks --out sim-out        # tracking, adds "+ VLM camera"
```

The camera is 640 × 480 with a 60° vertical field of view, mounted 1.3 m
high and running at 2 Hz. The people are a textured CC0 human mesh. Cars and
bicycles are built from boxes and cylinders, and every person is the same
mesh.

## Speed

The measurements below use a 4-core Sapphire Rapids VM.

| Model | Precision | Per frame |
| --- | --- | --- |
| Florence-2-base, one prompt | float32 | 2.5 s |
| Florence-2-base, one prompt | bfloat16 (AMX) | 1.0 s |
| Florence-2-base, detection + grounding (used) | bfloat16 | 1.7–2.2 s |
| Florence-2-large, one prompt | bfloat16 | 3–4 s |
| SmolVLM-256M, one question, one 512 px view | bfloat16 | 0.4–0.7 s |
| SmolVLM-256M, one question, image split into tiles | bfloat16 | 3.9–6.6 s |

bfloat16 gives the same boxes as float32 on the tested frames.

## 2D detection

`vlm_detect.py` asks Florence-2 two things per frame:

1. **`<OD>`** (open detection), used for people. It labels cars and bicycles
   alike as "land vehicle". A first version mapped only "car" and "bicycle",
   dropped every "land vehicle", and so found 3 % of the cars.
2. **Phrase grounding** with "a car. a bicycle.". It tells cars from bicycles,
   but it returns a box for every phrase even when nothing is there: on
   people-only scenes it invented vehicles and cut class accuracy to 0.49–0.76.
   A grounded vehicle is therefore kept only where `<OD>` also saw a vehicle
   ("land vehicle", a wheel, ...).

A person box that reaches down onto a bicycle box becomes one cyclist box. A
detection matches a truth box at IoU ≥ 0.5.

| task | truth boxes | recall | precision | class accuracy | recall < 40 px | 40-100 px | >= 100 px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| people_basic | 81 | 0.975 | 1.000 | 1.000 | - | 0.968 | 1.000 |
| crowd_crossing | 241 | 0.664 | 0.856 | 1.000 | - | 0.594 | 0.750 |
| occlusion | 83 | 0.916 | 0.974 | 1.000 | - | 0.899 | 1.000 |
| maneuvering | 112 | 0.964 | 1.000 | 1.000 | - | 0.951 | 1.000 |
| moving_robot | 162 | 0.944 | 0.994 | 1.000 | 0.944 | 0.824 | 0.982 |
| pose_jitter | 162 | 0.944 | 0.994 | 1.000 | 0.944 | 0.824 | 0.982 |
| mixed_traffic | 142 | 0.810 | 0.958 | 0.878 | 0.500 | 0.900 | 1.000 |
| long_range | 135 | 0.756 | 0.803 | 1.000 | 0.684 | 0.946 | - |
| side_by_side | 150 | 0.813 | 0.992 | 1.000 | 0.000 | 0.784 | 0.894 |

In mixed traffic, recall by class is 0.94 for people, 0.77 for cyclists and
0.56 for cars. Small distant cars are lost because `<OD>` does not confirm
them. The weak tasks are the crowd, where people overlap in the image, and
pairs that look like one figure at a distance (side_by_side, < 40 px).

## VLM detections in the tracker

In camera-assisted mode the tracker projects each confirmed LiDAR track's 3D
box into the image and matches it to the VLM boxes (IoU ≥ 0.2). Matches add
class evidence. The camera never creates, moves or deletes a track
([`docs/camera-lidar.md`](camera-lidar.md)).

| task | LiDAR only: HOTA / IDF1 / class accuracy | + VLM camera |
| --- | --- | --- |
| the 8 people-only tasks | unchanged | unchanged |
| mixed_traffic | 0.514 / 0.530 / 0.766 | 0.514 / 0.531 / 0.758 |

Association works: in mixed traffic 91 of 94 VLM detections matched a track.
It does not help because the camera can only weigh in on tracks the LiDAR
already made:

- **People** were already right from the LiDAR. Of the 469 matched
  track-scans, 380 carry a camera prior, and all 380 are right.
- **Cyclists**: the 26 priors all say "person". The rider's box overlaps the
  projected box better than the cyclist box does.
- **Cars**: none of the 268 matched car track-scans gets a prior. The LiDAR
  detector splits a car into person-sized pieces (see `docs/simulation.md`),
  and the camera, at 2 Hz, gives one box to one piece. No piece collects the 3
  hits within 1.5 s that a prior needs.

So the camera has to act before tracking to fix these errors: merge the LiDAR
clusters that project into one VLM car box, and label the merged detection
with the VLM class. That is the next step.

## Scene questions

`vlm_qa.py` asks SmolVLM-256M two questions about each frame (330 frames, 8
tasks, without image tiling):

- **count**: "How many people are in this image?" The truth counts people
  with at least 300 visible pixels.
- **path**: "Is there a person standing in the path less than 5 meters
  ahead?" The truth is a person 0–5 m ahead within 1 m of the heading line.

| | exact count | path recall | path false "yes" | path accuracy |
| --- | --- | --- | --- | --- |
| SmolVLM-256M | 0.48 | 0.32 | 0.39 | 0.58 |
| Florence-2 person boxes, counted | 0.67 | | | |

Per task:

| task | frames | count exact (VLM) | count MAE (VLM) | count exact (Florence boxes) | count MAE (Florence boxes) | path accuracy | path frames with a person | path recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| people_basic | 40 | 0.80 | 0.20 | 0.97 | 0.03 | 0.70 | 3 | 0.33 |
| crowd_crossing | 40 | 0.53 | 2.30 | 0.57 | 1.00 | 0.62 | 0 | - |
| occlusion | 40 | 0.28 | 0.93 | 0.90 | 0.10 | 0.47 | 0 | - |
| maneuvering | 40 | 0.70 | 0.33 | 0.82 | 0.17 | 0.42 | 3 | 0.33 |
| moving_robot | 50 | 0.52 | 0.60 | 0.62 | 0.50 | 0.28 | 22 | 0.32 |
| mixed_traffic | 40 | 0.42 | 0.70 | 0.57 | 0.50 | 0.85 | 0 | - |
| long_range | 40 | 0.17 | 2.30 | 0.17 | 2.70 | 0.97 | 0 | - |
| side_by_side | 40 | 0.42 | 0.70 | 0.72 | 0.33 | 0.42 | 0 | - |

A 256 M chat model does not answer these reliably. It counts worse than
Florence-2's boxes, and its path answers are about chance. "Is the path
clear?" should come from geometry (tracks, or detections with depth), not
from a small VLM's yes/no. A larger chat VLM (2–3 B parameters) would take
roughly 5–10 s per question on this CPU and has not been tried. The long-range
counts disagree for another reason: the truth leaves out people under 300
pixels, but both models count them.
