# Simulated perception tasks

Real datasets mix every difficulty in every window. The simulated tasks
separate them, one scene per difficulty, with exact truth. The scenes come
from Robot Native Engine's
[example 149](https://github.com/rsasaki0109/RobotNativeEngine/tree/main/examples/149_perception_tasks):

- a 16-channel LiDAR scans moving people, cyclists and cars among walls,
  pillars and poles;
- a cluster detector turns each scan into detections and pole landmarks;
- the results are written in this repository's replay format (`setup.json`,
  `inputs.jsonl`, `truth.jsonl`).

The detector is geometric. Its failures (merged pairs, split cars, cyclists
labelled as people) are part of each task.

## Reproduce

```bash
# in RobotNativeEngine
cargo run --release -p perception_tasks --example 149_perception_tasks -- --out /tmp/perception_tasks
# here
cargo build --release -p robot-perception
python3 scripts/sim/evaluate_tasks.py /tmp/perception_tasks --out sim-out --json sim-out/results.json
```

Simulation takes about 17 s and tracking plus scoring about 20 s (numpy and
scipy are needed).

## Tasks

| Task | Scene | What it tests |
| --- | --- | --- |
| `people_basic` | 4 people walking across, toward and away from a standing robot | Baseline |
| `crowd_crossing` | Two groups of 7 people pass through each other | Identity through a crowd |
| `occlusion` | 5 people pass behind pillars, a kiosk and each other | Full occlusions |
| `maneuvering` | Stop-and-go, a U-turn, a 3 m/s jogger, a zig-zag, a start from standing | Constant-velocity model |
| `moving_robot` | The robot drives down a corridor at 1 m/s among 6 people | Ego motion |
| `pose_jitter` | As `moving_robot` with 1° / 0.1 m of ego-pose noise per scan | Pose correction |
| `mixed_traffic` | Pedestrians, cyclists and cars (7–9 m/s) | Classes, fast objects, detector failures |
| `long_range` | People 12–35 m away, about 6 points each | Sparse returns |
| `side_by_side` | Pairs 0.6 m apart; one pair splits up | Merged detections |

## Results

All tracking runs are LiDAR-only. They use the ROS 2 node's people defaults:
`--max-association-distance-m 1.0 --recent-track-window-ns 0`. Only confirmed
tracks are scored, matched within 1 m (2 m in `mixed_traffic`, where car
centres are off by up to a metre).

Detector recall and precision are measured before tracking. Velocity is the
median error against the truth's velocity.

| task | tracker | detection recall | detection precision | HOTA | IDF1 | MOTA | ID switches | velocity median (m/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| people_basic | tracker | 0.979 | 1.000 | 0.798 | 0.975 | 0.950 | 0 | 0.042 |
| people_basic | + 0.5 s smoothing | 0.979 | 1.000 | 0.803 | 0.975 | 0.950 | 0 | 0.026 |
| crowd_crossing | tracker | 0.807 | 1.000 | 0.644 | 0.846 | 0.854 | 6 | 0.062 |
| crowd_crossing | + 0.5 s smoothing | 0.807 | 1.000 | 0.647 | 0.847 | 0.855 | 5 | 0.044 |
| occlusion | tracker | 0.674 | 1.000 | 0.577 | 0.718 | 0.707 | 3 | 0.066 |
| occlusion | + 0.5 s smoothing | 0.674 | 1.000 | 0.583 | 0.718 | 0.707 | 3 | 0.045 |
| maneuvering | tracker | 0.949 | 1.000 | 0.789 | 0.978 | 0.956 | 0 | 0.058 |
| maneuvering | + 0.5 s smoothing | 0.949 | 1.000 | 0.795 | 0.978 | 0.956 | 0 | 0.035 |
| moving_robot | tracker | 0.967 | 0.985 | 0.803 | 0.981 | 0.962 | 0 | 0.046 |
| moving_robot | + 0.5 s smoothing | 0.967 | 0.985 | 0.807 | 0.981 | 0.962 | 0 | 0.027 |
| moving_robot | + pose correction | 0.967 | 0.985 | 0.802 | 0.981 | 0.962 | 0 | 0.068 |
| pose_jitter | tracker | 0.967 | 0.984 | 0.702 | 0.902 | 0.911 | 1 | 0.294 |
| pose_jitter | + 0.5 s smoothing | 0.967 | 0.984 | 0.732 | 0.902 | 0.911 | 1 | 0.165 |
| pose_jitter | + pose correction | 0.967 | 0.984 | 0.696 | 0.980 | 0.961 | 0 | 0.096 |
| mixed_traffic | tracker | 0.631 | 0.556 | 0.514 | 0.530 | 0.275 | 45 | 0.108 |
| mixed_traffic | + 0.5 s smoothing | 0.631 | 0.556 | 0.516 | 0.533 | 0.255 | 45 | 0.083 |
| long_range | tracker | 0.889 | 1.000 | 0.673 | 0.857 | 0.742 | 0 | 0.046 |
| long_range | + 0.5 s smoothing | 0.889 | 1.000 | 0.682 | 0.857 | 0.742 | 0 | 0.029 |
| side_by_side | tracker | 0.845 | 0.993 | 0.715 | 0.849 | 0.877 | 3 | 0.046 |
| side_by_side | + 0.5 s smoothing | 0.845 | 0.993 | 0.720 | 0.849 | 0.877 | 3 | 0.029 |

## What the tasks show

- **Smoothing** (`--smooth-lag 5`) lowers the velocity error by 30–45 % in
  every task. On CODa the reduction was 23–29 %. It barely changes identity.
- **Pose correction** matters only when the poses jitter. In `pose_jitter` it
  takes IDF1 from 0.902 to 0.980 and MOTA from 0.911 to 0.961. The velocity
  error falls from 0.29 to 0.10 m/s. HOTA stays level: the localisation term
  scores the corrected frame, and that frame carries the residual alignment
  error. With clean poses (`moving_robot`) it costs a little velocity
  accuracy, as on CODa's held-out windows.
- **Occlusion** is the weakest people task (HOTA 0.58). The losses are mostly
  missed detections while people are hidden (recall 0.67), not wrong
  identities.
- **Crowds and pairs**: the 6 switches in `crowd_crossing` and 3 in
  `side_by_side` happen where two people merge into one cluster.
- **Manoeuvres** cause no switches. Stops, U-turns and a 3 m/s jogger stay
  inside the 1 m gate at 10 Hz.
- **Mixed traffic** is limited by the detector, not by association. Widening
  the gate to 2–3 m cuts the switches only from 45 to 35, and false positives
  stay above 600: they come from cars split by occlusion and from cyclists
  that look like people end-on. This is the same conclusion as on CODa
  (`docs/evaluation.md`): detection, not association, has the most headroom.

## Closed loop on a Go2

The tasks above score tracks. Robot Native Engine's
[example 151](https://github.com/rsasaki0109/RobotNativeEngine/tree/main/examples/151_go2_people)
scores what a robot does with them instead. A Go2 sees people only through
its Mid-360. It follows a person, yields to people crossing, and passes
people in a corridor. The `stream` command runs inside its control loop, one
`lidar_batch` in and one `lidar_frame` out per 0.1 s scan, at about 0.1 ms
per frame:

```bash
cargo run --release -p go2_people --example 151_go2_people -- --out target/go2_people --tracker \
    /path/to/robot-perception stream --setup target/go2_people/stream_setup.json \
    --max-association-distance-m 1.0 --recent-track-window-ns 0
```

Following and crossing succeed with every source. In the corridor:

Eight trials per task (each person starting up to 2 s earlier or later),
runs with a contact out of 8:

| perception | follow | crossing | corridor | furnished room |
| --- | --- | --- | --- | --- |
| truth | 0 | 0 | 0 | 0 |
| robot-perception | 0 | 0 | 4 | 2 |
| robot-perception + static map | 0 | 0 | 2 | 0 |
| robot-perception + static map + SLAM | 0 | 0 | 2 | 3 |
| greedy nearest neighbour | 1 | 0 | 3 | 2 |
| greedy nearest neighbour + static map | 1 | 0 | 6 | 0 |

robot-perception is on the wrong person 6 % of the time when following,
against 12 % for the greedy tracker. Wall pieces and furniture pass the
simple person test and become phantom people. A static map built online from
the robot's own scans removes most of them. SLAM localizes the robot to 4 cm
in the furnished room but does not yet make it safer there; see the example's
write-up.

## Camera and VLM

With `--camera`, the tasks also come with a CPU-rendered forward camera
and exact 2D truth. Florence-2 (a pretrained VLM, on CPU) detects people,
cars and cyclists in those frames, and its detections feed the tracker's
camera-assisted mode. A small chat VLM answers scene questions. Results and
what limits them: [`docs/vlm.md`](vlm.md).

## Limits

- People are rigid capsules: no limbs or gait.
- Movers pass through each other.
- Scans are instantaneous.
- The ground is flat and known.
- Camera people are one mesh in a fixed standing pose, so appearance cues
  (re-identification) are not covered.
- Scores depend on the simple detector. They compare tracker settings on the
  same detections; they are not absolute performance.
