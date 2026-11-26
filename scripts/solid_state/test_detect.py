import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from detect import (cluster_predictor, mid360_points, pedestrian_detections,
                    realsense_points, replace_detections)


class DetectorAdapterTests(unittest.TestCase):
    def test_mid360_xyz_is_preserved_and_intensity_is_scaled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000000.bin"
            np.array([[1, 2, 3, 255], [1, float("nan"), 3, 0]], dtype="<f4").tofile(path)
            np.testing.assert_allclose(mid360_points(path, 1 / 255), [[1, 2, 3, 1]])
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "packed float32"):
                mid360_points(path, 1.0)

    def test_realsense_depth_uses_calibrated_axes_and_scale(self):
        calibration = {
            "fx": 2, "fy": 2, "cx": 0, "cy": 0, "depth_scale_m": 0.001,
            "optical_to_input": [[0, 0, 1, 0.1], [-1, 0, 0, 0],
                                 [0, -1, 0, 0.4], [0, 0, 0, 1]],
        }
        points = realsense_points(np.array([[0, 1000], [2000, 0]]), calibration)
        np.testing.assert_allclose(points, [[1.1, -0.5, 0.4, 0], [2.1, 0, -0.6, 0]])
        calibration["fx"] = 0
        with self.assertRaises(ValueError):
            realsense_points(np.ones((1, 1)), calibration)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "cluster backend requires SciPy")
    def test_cluster_detects_person_size_returns_and_ignores_floor(self):
        predict = cluster_predictor(SimpleNamespace(floor_z_m=-0.5, score_min=0.3))
        points = np.array([[1.02, 0.01, -0.2, 0], [1.04, 0.03, 0.2, 0],
                           [1.08, 0.08, 0.8, 0], [1.1, 0.1, 1.0, 0],
                           [1.1, 0.1, -0.49, 0]])
        detections = predict(points)
        self.assertEqual(len(detections), 1)
        np.testing.assert_allclose(detections[0]["position"], [1.06, 0.055, 0.4])

    def test_pedestrian_boxes_and_record_metadata(self):
        detections = pedestrian_detections(
            [[2, 0, 0.3, 2, 1, 2, np.pi / 2], [3, 0, 1, 1, 1, 1, 0]],
            [0.9, 0.8], [2, 1], 2, 0.5, -0.5,
        )
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection["class"], 2)
        np.testing.assert_allclose(detection["position"], [2, 0, 0.8])
        corners = np.array(detection["shape"]["corners"]).reshape(8, 3)
        np.testing.assert_allclose(np.ptp(corners, axis=0), [1, 2, 2])
        with tempfile.TemporaryDirectory() as directory:
            source, output = (Path(directory) / name for name in ("in.jsonl", "out.jsonl"))
            lidar = {"type": "lidar_batch", "sensor_id": 1, "sequence": 4,
                     "event_time_ns": 13, "arrival_time_ns": 17, "frame_id": "base",
                     "detections": ["old"]}
            camera = {"type": "camera_batch", "sensor_id": 2, "detections": ["keep"]}
            source.write_text(json.dumps(lidar) + "\n" + json.dumps(camera) + "\n")
            self.assertEqual(replace_detections(source, output, 1, lambda seq: seq,
                                                lambda seq: detections if seq == 4 else []), 1)
            self.assertEqual(list(map(json.loads, output.read_text().splitlines())),
                             [dict(lidar, detections=detections), camera])
            original = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "no lidar_batch"):
                replace_detections(source, output, 3, lambda _: None, lambda _: [])
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
