import unittest

import numpy as np

from pair_lidars import common_mask


class CommonViewTest(unittest.TestCase):
    def test_uses_both_sensor_poses_and_base_range(self):
        mid_from_base = np.eye(4)
        xt_from_base = np.eye(4)
        xt_from_base[2, 3] = -0.5
        points = np.array([[2, 0, 0], [1, 0, 0], [3, 0, 2], [9, 0, 0], [0.1, 0, 0]])
        np.testing.assert_array_equal(common_mask(points, (mid_from_base, xt_from_base)),
                                      [True, False, False, False, False])


if __name__ == "__main__":
    unittest.main()
