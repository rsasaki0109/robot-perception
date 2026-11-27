import unittest

import numpy as np

from score_camera_labels import distortion_for, project_pixels, score


class CameraLabelScoringTest(unittest.TestCase):
    def test_rectified_projection_and_behind_camera_mask(self):
        intrinsics = np.array([[4, 0, 50], [0, 6, 40], [0, 0, 1]])
        distortion = distortion_for(dict(distortion_model="plumb_bob", D=[0] * 5))
        pixels, front = project_pixels(
            np.array([[2, -1, 2], [0, 0, -1]]), np.eye(4), intrinsics,
            distortion, np.array([0.5, 0.5]),
        )
        np.testing.assert_allclose(pixels, [[27, 18.5]])
        np.testing.assert_array_equal(front, [True, False])

    def test_matching_and_ignored_prediction(self):
        predictions = np.array([[0, 0, 4, 4], [5, 5, 7, 7]])
        labels = np.array([[0, 0, 4, 8]])
        ignore = np.array([[5, 5, 8, 8]])
        self.assertEqual(score(predictions, labels, ignore, 0.50), ((1, 0, 0), {0}))
        self.assertEqual(score(predictions, labels, ignore, 0.51), ((0, 1, 1), set()))


if __name__ == "__main__":
    unittest.main()
