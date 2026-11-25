import unittest

import numpy as np

from src.gesture import classify_hand, count_gestures, gesture_satisfied


def _hand() -> np.ndarray:
    return np.zeros((21, 2), dtype=np.float32)


def _peace() -> np.ndarray:
    points = _hand()
    points[8] = (0.0, -2.0)
    points[6] = (0.0, -1.0)
    points[12] = (0.3, -2.0)
    points[10] = (0.3, -1.0)
    points[9] = (0.3, -0.4)
    points[16] = (0.6, -0.3)
    points[14] = (0.6, -0.9)
    points[20] = (0.9, -0.2)
    points[18] = (0.9, -0.8)
    points[4] = (-0.4, -0.4)
    return points


def _heart() -> np.ndarray:
    points = _hand()
    points[9] = (0.0, -1.0)
    points[4] = (0.15, -0.9)
    points[8] = (0.20, -0.9)
    points[12] = (0.0, -2.2)
    points[10] = (0.0, -1.3)
    points[16] = (0.4, -2.1)
    points[14] = (0.4, -1.2)
    points[20] = (0.7, -2.0)
    points[18] = (0.7, -1.2)
    return points


def _open_palm() -> np.ndarray:
    points = _peace()
    points[16] = (0.6, -2.0)
    points[14] = (0.6, -1.0)
    points[20] = (0.9, -1.9)
    points[18] = (0.9, -1.0)
    return points


class GestureTest(unittest.TestCase):
    def test_peace(self):
        self.assertEqual(classify_hand(_peace()), "peace")

    def test_heart(self):
        self.assertEqual(classify_hand(_heart()), "finger_heart")

    def test_open_palm_is_not_a_pose(self):
        self.assertIsNone(classify_hand(_open_palm()))

    def test_counts_and_requirements(self):
        peace, heart = count_gestures([_peace(), _peace(), _heart()])
        self.assertEqual((peace, heart), (2, 1))
        self.assertTrue(gesture_satisfied("peace_two", peace, heart))
        self.assertFalse(gesture_satisfied("peace_one", peace, heart))
        self.assertTrue(gesture_satisfied("finger_heart", 0, 1))
        self.assertTrue(gesture_satisfied("peace_one", 1, 0))
        self.assertFalse(gesture_satisfied("peace_one", 2, 0))


if __name__ == "__main__":
    unittest.main()
