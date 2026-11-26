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
