# Pedestrian detectors for Mid-360 and RealSense

The [detector adapter](../scripts/solid_state/detect.py) runs OpenPCDet
and writes pedestrian detections into one input sensor's JSONL batches.
It keeps sequence IDs, sensor timestamps, frame IDs, and other streams,
so detector comparisons can use the same tracker and calibration.

## Mid-360 on GrandTour ETH-2

Prepare the matching raw GrandTour recording with
[`grandtour.py prepare`](mid360.md). The benchmark includes the JSONL
template, calibration and identity annotations. Its raw scans come from
GrandTour. Use the same scan IDs in the template and the prepared folder.
The exported clouds are already in the robot base frame, including the
transform for the upside-down Mid-360 mount.

```sh
repo=/path/to/robot-perception
sequence=/path/to/prepared/eth2
gzip -dc "$repo/benchmarks/grandtour_eth2/inputs.jsonl.gz" > "$sequence/baseline.jsonl"

cd /path/to/OpenPCDet/tools
python "$repo/scripts/solid_state/detect.py" \
  --sensor mid360 --scans "$sequence/scans" \
  --template "$sequence/baseline.jsonl" --out "$sequence/pointpillars.jsonl" \
  --cfg cfgs/kitti_models/pointpillar.yaml --ckpt /path/to/pointpillars.pth \
  --intensity-scale 0.003921568627451
```

Run from the matching OpenPCDet checkout's `tools` directory with PyTorch,
CUDA and the compiled OpenPCDet extensions. The config and checkpoint must
match and include the `Pedestrian` class. Use the same command with the
matching pedestrian CenterPoint config and checkpoint to compare a second
architecture. The [existing CODa CenterPoint recipe](detector_training.md)
uses its own OpenPCDet fork and checkout. Keep each model with the runtime
it was trained for.

Size the input crop and match feature scaling to each checkpoint. The example
divides raw Livox reflectivity by 255. Pass a model-specific `--z-offset-m`
if the detector expects a different floor height. The adapter reverses that
offset for output boxes. A KITTI checkpoint is a transfer baseline with a
front-facing, spinning-LiDAR training distribution. Its output does not
establish full-surround Mid-360 coverage.

## Run the published CODa CenterPoint on CPU

The `coda-cpu` backend uses the existing
[CODa CPU detector](../scripts/coda/detect_lidar_cpu.py) and its built-in
configuration. Install CPU PyTorch 2.5.1 or newer, NumPy and Shapely. The
[pedestrian checkpoint](https://huggingface.co/rsasaki0109/coda-ped-centerpoint)
used for ETH-2 has SHA-256
`8fe8ac00fcde6c28165834b0a4a919f53683ed8911e91ae2e1bd2e912072463b`.

```sh
python "$repo/scripts/solid_state/detect.py" \
  --backend coda-cpu --sensor mid360 --scans "$sequence/scans" \
  --template "$sequence/baseline.jsonl" --out "$sequence/centerpoint.jsonl" \
  --ckpt /path/to/coda_ped_centerpoint_ft.pth \
  --z-offset-m -0.25 --score-min 0.5 --nms-threshold 0.3 --cpu-threads 1
```

This checkpoint uses XYZ features. The CPU path runs without CUDA and
uses the model's own voxel geometry and rotated NMS. The PointPillars
command above uses the CUDA backend.

## RealSense depth recordings

Save each **rectified** depth image as a 2D NumPy array named by its
six-digit sequence ID, such as `000004.npy`. Use the camera's depth scale
and rectified intrinsics. The calibration file also needs a row-major 4×4
`optical_to_input` transform from optical axes (right, down, forward)
to the frame named in the JSONL input record. The example below puts the
same origin in forward-left-up axes. Replace the intrinsics, depth scale,
and mount transform with those from the recording.

```json
{
  "fx": 600.0, "fy": 600.0, "cx": 319.5, "cy": 239.5,
  "depth_scale_m": 0.001,
  "optical_to_input": [[0,0,1,0], [-1,0,0,0], [0,-1,0,0], [0,0,0,1]]
}
```

Use one placeholder batch per recorded depth frame, with matching
timestamps and sequence IDs. For example:

```json
{"type":"lidar_batch","schema_version":1,"sensor_id":1,"sequence":4,"event_time_ns":100000000,"arrival_time_ns":100000000,"frame_id":"base","detections":[]}
```

Run the same detector command with `--sensor realsense --scans /path/to/depth
--calibration /path/to/depth_calibration.json`, and supply the matching
stream template and output. RealSense is an RGB-D camera. This path uses
its depth as geometry and fills the input intensity channel with zero.
A depth-domain checkpoint is needed for an accuracy claim. The adapter
does not capture a live device or consume an unrectified recording.

## Replay and compare

```sh
factor-tracker replay --mode lidar_only --setup "$sequence/setup.json" \
  --inputs "$sequence/pointpillars.jsonl" --output "$sequence/tracks.jsonl" \
  --diagnostics "$sequence/diagnostics.json"
```

Keep timestamps, tracker settings and held-out sequences fixed when comparing
learned detectors with existing detections. The supplied ETH-2 identity
annotations start from existing detector boxes. They can help compare
tracking on annotated people, but cannot measure missed people independently.
For detection recall and distance/occlusion breakdowns, use independent
labels. Keep Mid-360 and RealSense results separate because range, field of
view, sampling and return features differ. See the
[ETH-2 benchmark](../benchmarks/grandtour_eth2/README.md).

## Real-recording validation on GrandTour ETH-2

Used the [public mission `2024-10-01-11-47-44`](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/tree/af6b6dd6f91696f8ec8c98134e195514834911b2/2024-10-01-11-47-44)
and the published pedestrian CenterPoint checkpoint above. Mid-360 used
scans 3700–3900 inclusive. Each cloud combines three undistorted sweeps
in the current base frame, with the recorded mount and interpolated
odometry as in `grandtour.py`. Scan timestamps matched the bundled ETH-2
records. Both replays used LiDAR only, the same setup and the benchmark's
default tracker with initial velocity variance 1, acceleration spectral
density 0.5 and evaluation match distance 1 m.

| Mid-360 input, 201 scans | Detection boxes | Matched annotated objects | Track coverage | ID switches |
| --- | ---: | ---: | ---: | ---: |
| Bundled PV-RCNN detections | 3,740 | 2,826 / 2,855 | 99.0% | 22 |
| Fresh CenterPoint inference, score ≥ 0.5 | 3,066 | 2,317 / 2,855 | 81.2% | 97 |

The reference annotations are derived from the bundled detector, so the
comparison measures tracking against those annotations and favors the
bundled detections. The CPU backend took 1.114 s median and 1.249 s p95
per Mid-360 scan on one PyTorch thread. The timings cover preprocessing,
model execution and box conversion, excluding file I/O and tracking.
That does not meet the sensor's 100 ms scan interval.

For the RealSense smoke run, the same mission's front-upper D435i stream
provided 12 frames (5549–5560). We decoded the 848×480 uint16 PNGs as
millimetres and used the recording's zero-distortion intrinsics and inverse
of its base-to-optical mount transform. The dominant floor bin was
−0.50 m in both base-frame point clouds near the start of the window.
CenterPoint produced no detections at score ≥ 0.5 (CPU median 1.343 s).
At score ≥ 0.1 it produced nine low-score boxes, with max score 0.201 and
zero confirmed tracks. All 12 frames replayed at both thresholds.
This checks the recorded-depth path and the spinning-LiDAR checkpoint's
behavior on that clip. It does not establish RealSense detection accuracy.
The [sensor setup](https://grand-tour.leggedrobotics.com/setup) describes
the rig.


## One sweep versus three on the same scans

All rows use ETH-2 scan IDs 3700–3900 and the same tracker settings and
matching criterion above. A single-sweep file uses only the current
LiDAR packet after mount transformation. Three sweeps also use the
previous two scans, aligned by odometry. CenterPoint kept the same
checkpoint, 0.5 score and 0.3 rotated NMS thresholds for both inputs.

| Detector | Sweeps | Boxes | Track coverage | ID switches | Detector median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bundled PV-RCNN detections | 3 | 3,740 | 99.0% | 22 | — |
| CenterPoint CPU | 1 | 2,441 | 81.5% | 89 | 1038 / 1162 ms |
| CenterPoint CPU | 3 | 3,066 | 81.2% | 97 | 1114 / 1249 ms |
| Geometric clustering | 1 | 6,638 | 79.8% | 82 | 2.4 / 3.3 ms |
| Geometric clustering | 3 | 6,022 | 65.1% | 83 | 4.5 / 7.0 ms |

The clustering backend ports example 151's scan-only connected-cell
classifier to this adapter. It uses 0.15 m cells, a 15 m radial crop and
a fixed floor at −0.50 m in the ANYmal base frame. Its score of 1 is
a placeholder. Timings exclude disk reads and tracking. These coverage
values use detector-derived annotations. Box counts include detections
outside the annotated area, so neither column gives recall or false
positive rate. The supplied PV-RCNN results cover the three-sweep
condition. PointPillars has yet to be run on CUDA.

```sh
python "$repo/scripts/solid_state/detect.py" \
  --backend cluster --sensor mid360 --floor-z-m -0.5 \
  --scans "$sequence/scans" --template "$sequence/baseline.jsonl" --out "$sequence/cluster.jsonl"
```

### Detector-blind review

The review generator selects every twentieth scan from 3700 to 3900
(11 sheets), pairs each with the same-index front image roughly 48 ms
later and shows single- and three-sweep point clouds without detector
overlays. It initializes a per-frame review checklist and an empty
person annotation CSV. Mark the reviewed region, occlusion and uncertain
cases explicitly. Use adjacent scans to judge motion. The camera only
confirms its field of view. A reviewer can record people that none of
the detectors returned. Human review and the independent distance and
occlusion breakdowns are pending.

```sh
python "$repo/scripts/solid_state/review.py" \
  --single /path/to/single-sweeps --three "$sequence/scans" \
  --images /path/to/raw/alphasense_front_center --out "$sequence/independent-review"
```

Export the one-sweep files with the same GrandTour preparation and
`--accumulate 1`, and the three-sweep files with `--accumulate 3`.
Use the raw JPEGs for review. The plots require Matplotlib, NumPy and Pillow.

### RealSense with an RGB person detector

People are visible in the same mission's front camera around the D435i
clip. In a CPU pilot on six raw Alphasense frames (3700–3750, step 10),
COCO-trained SSDLite MobileNetV3 produced six person boxes with scores
≥ 0.5 across four images. For image 3700, the D435i frame 5549 is
24.9 ms earlier. Projecting the depth points through the recorded mounts
and Alphasense fisheye intrinsics gave depth samples inside the central
half of both RGB person boxes. Their median planar ranges were 3.61 m
and 5.49 m. The pilot uses the shared base frame without compensating
the 24.9 ms offset. The ranges are unvalidated samples and may include
background. It did not use the LiDAR checkpoint or measure RGB-D
recognition accuracy.

## Paired Mid-360 and Hesai XT32 on ETH-2

The same ANYmal D and Boxi payload recorded both sensors. Use the pinned
[ETH-2 mission](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/tree/af6b6dd6f91696f8ec8c98134e195514834911b2/2024-10-01-11-47-44/data).
Unpack `livox_points_undistorted.tar`, `hesai_points_undistorted.tar`,
`anymal_state_odometry.tar`, and `tf.tar` into the Zarr `data` directory
with the mission's `.zgroup` file.

```sh
python scripts/grandtour/pair_lidars.py --data /path/to/mission/data \
  --template eth2/inputs.jsonl --truth eth2/truth.jsonl --out eth2/paired
```

The pairing script uses the nearest unused XT32 scan within 50 ms, keeps
each sensor's single undistorted sweep, and applies the calibrated mounts
and odometry to express both sweeps in the Mid-360 scan's base frame. It
clips points and annotated person centers to 0.5–8 m planar range and to the
intersection of the sensors' nominal elevation intervals in their own
frames (Mid-360 −7° to +52°, XT32 −16° to +15°). The result has 201 pairs.
The XT32 timestamps precede Mid-360 by 26.8 ms median (absolute p95 27.1 ms,
max 27.8 ms). Odometry accounts for robot motion. Pedestrians can move
between acquisitions, and the angular crop cannot rule out occlusion.
See the [rig calibration](https://grand-tour.leggedrobotics.com/setup)
and [XT32 manual](https://hesai.pousion.cn/wp-content/uploads/2023/03/XT32_User_Manual_X01-en-220410.pdf).

Use `--scans eth2/paired/mid360 --sensor mid360` or
`--scans eth2/paired/xt32 --sensor xt32` with the detector and the paired
`inputs.jsonl` template. Reuse the original ETH-2 `setup.json` and score
against the paired `truth.jsonl`. The crop retained all 2,855 original
annotation instances. Both sensors use the same tracker settings as the
previous comparison. The cluster baseline uses the same −0.50 m floor
assumption and the CPU CenterPoint backend uses XYZ only, with the same
checkpoint, z offset, score threshold, and NMS.

| Sensor | Detector | Median points per cropped sweep | Boxes | Track coverage | ID switches | Detector median / p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mid-360 | Cluster | 6,300 | 4,330 | 80.0% | 76 | 1.7 / 2.0 ms |
| XT32 | Cluster | 36,205 | 3,879 | 63.0% | 102 | 4.6 / 6.2 ms |
| Mid-360 | CenterPoint CPU | 6,300 | 1,648 | 67.7% | 56 | 962 / 1,078 ms |
| XT32 | CenterPoint CPU | 36,205 | 2,295 | 60.8% | 101 | 999 / 1,090 ms |

The XT32 had 5.7 times as many points in the crop. Its track coverage
against the supplied annotations was lower for both detector settings.
The labels were built from a Mid-360 baseline, and the two sensor mounts
and scan patterns expose different surfaces. These numbers measure
transfer under fixed detector settings on this clip. They cannot establish
which hardware has better person-detection recall.

### Camera-only annotation pilot

An AI reviewer drew visible-person boxes from three raw
`alphasense_front_center` images (3700, 3740, 3780) before inspecting detector
outputs. The [annotations](../benchmarks/grandtour_eth2/camera_labels_pilot.json)
contain 19 person instances in the front-camera region of interest, with a
minimum visible height of 120 px at 960 × 720. Small distant figures and
people whose outlines were difficult to separate were marked as ignore regions.

Unpack `alphasense_front_center.tar` into the same mission data directory.
Use the detector's LiDAR batch JSONL files from the paired runs.

```sh
python scripts/grandtour/score_camera_labels.py --data /path/to/mission/data \
  --labels benchmarks/grandtour_eth2/camera_labels_pilot.json \
  --setup benchmarks/grandtour_eth2/setup.json.gz \
  eth2/paired/mid360-cluster.jsonl eth2/paired/mid360-centerpoint.jsonl \
  eth2/paired/xt32-cluster.jsonl eth2/paired/xt32-centerpoint.jsonl
```

The scorer transforms each 3D box from the paired Mid-360 base frame to
the camera frame at exposure time using robot odometry, then projects its
corners with the raw fisheye intrinsics. It skips boxes with corners at or
behind the camera plane and boxes whose projected center falls outside the ROI.
It matches boxes one-to-one at 2D IoU ≥ 0.30. An unmatched prediction whose
center falls in an ignore region does not count as a false positive.

| Sensor | Detector | Matched | Unmatched predictions | Missed labels | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Mid-360 | Cluster | 12 | 3 | 7 | 63.2% |
| XT32 | Cluster | 8 | 1 | 11 | 42.1% |
| Mid-360 | CenterPoint CPU | 13 | 0 | 6 | 68.4% |
| XT32 | CenterPoint CPU | 6 | 0 | 13 | 31.6% |

The three images can show the same people multiple times. The camera labels
have no independent range measurements, so some people can fall outside
the common LiDAR crop. Odometry compensates for robot motion between
timestamps, and pedestrians can still move between captures. These counts
measure agreement with visible 2D boxes on this clip. They do not estimate
full-field recall, 3D localization accuracy, or tracking quality.

### ETH-2 front-camera evaluation by range

The expanded review uses frames 3700 through 3900 at intervals of 20 scans
(about two seconds). An AI reviewer drew
[55 visible-person boxes](../benchmarks/grandtour_eth2/camera_labels_eth2.json)
across 11 images without detection overlays. The fixed 640 × 480 review
region is x = 65–582, y = 85–280. The visible-height cutoff is 80 px. Some
nearby boxes are clipped at the top of the review strip. Ambiguous figures
and distant groups have ignore regions. The review includes the three pilot
frames and repeatedly observes some of the same people in one crowded hallway.

For range estimates, project the synchronized D435i depth pixels into the
Alphasense image using the recorded intrinsics, mounts, and robot odometry.
Use the median planar range of pixels in the central 40% of each box along
each axis. Require at least 25 samples and a range interquartile spread of
at most 0.6 m. The depth frames lead the color frames by 23–25 ms. The
reference torso point must pass the same range and elevation masks as the
paired LiDAR data. This yielded 46 labels for common-crop range scoring,
one outside the crop, and eight without a reliable range. The method can
pick up an occluder or background surface, and a torso test does not
establish that both LiDARs saw the same body surfaces.

| Sensor | Detector | TP / FP / FN at IoU 0.30 | Precision | Recall | TP at IoU 0.50 |
| --- | --- | ---: | ---: | ---: | ---: |
| Mid-360 | Cluster | 24 / 22 / 31 | 52.2% | 43.6% | 22 |
| XT32 | Cluster | 18 / 19 / 37 | 48.6% | 32.7% | 15 |
| Mid-360 | CenterPoint CPU | 25 / 4 / 30 | 86.2% | 45.5% | 19 |
| XT32 | CenterPoint CPU | 23 / 1 / 32 | 95.8% | 41.8% | 20 |

| Sensor | Detector | 0.5–4 m | 4–6 m | 6–8 m | All 46 in common crop |
| --- | --- | ---: | ---: | ---: | ---: |
| Mid-360 | Cluster | 3/5 | 10/19 | 7/22 | 20/46 |
| XT32 | Cluster | 2/5 | 7/19 | 6/22 | 15/46 |
| Mid-360 | CenterPoint CPU | 2/5 | 5/19 | 17/22 | 24/46 |
| XT32 | CenterPoint CPU | 2/5 | 6/19 | 10/22 | 18/46 |

The range table counts matches at IoU 0.30. With CenterPoint, Mid-360 had
more matches across the common-crop labels, driven by the 6–8 m band.
XT32 had higher precision across all 55 labels and one more match when
the IoU threshold increased to 0.50. This single hallway clip supports a
comparison of the configured detectors in the reviewed front-camera
region. Full-azimuth and multi-mission results require annotated side
and rear observations and additional scenes.

Use calibrated D435i PNGs from the mission's `images/depth_camera_front_upper.tar`
and unpack that stream's metadata from its separate `data/` archive.
The [saved ranges](../benchmarks/grandtour_eth2/camera_ranges_eth2.json)
and [per-frame scores](../benchmarks/grandtour_eth2/camera_scores_eth2.json)
include the depth sample counts and both IoU thresholds.

```sh
python scripts/grandtour/camera_label_ranges.py --data /path/to/mission/data \
  --depth-images /path/to/mission/images/depth_camera_front_upper \
  --labels benchmarks/grandtour_eth2/camera_labels_eth2.json \
  --pairs eth2/paired/pairs.jsonl --out eth2/camera_ranges.json
python scripts/grandtour/score_camera_labels.py --data /path/to/mission/data \
  --labels eth2/camera_ranges.json --setup benchmarks/grandtour_eth2/setup.json.gz \
  --iou 0.30 eth2/paired/{mid360,xt32}-{cluster,centerpoint}.jsonl
```

## Unitree L2 official indoor sample

The [official L2 sample](https://www.unitree.com/download/L2/) contains
2,807 `/unilidar/cloud` PointCloud2 messages. With NumPy and `rosbags`
installed, inspect it and optionally export a scan as little-endian XYZI.

```sh
python scripts/solid_state/probe_l2.py "L2 Indoor Point Cloud Data.bag" \
  --export-first /path/to/000000.bin
```

The first 20 scans contain 5,243–5,341 finite returns per scan in
`unilidar_lidar`. The closest return was 0.095 m from the sensor and
19 of 20 scans had at least one return below 0.5 m. Those are return
distances, with no people labels or classification of near returns.
The PointCloud2 messages also carry ring and per-point time; the XYZI
export leaves those fields out. The sample's SHA-256 is
`2f1d7c01493299e8a310af3f881d615358c17331cac9b2426e18c8ca90149983`.
