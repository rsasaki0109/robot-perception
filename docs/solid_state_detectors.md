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
