import unittest

from src.decide import Observation
from src.expression import empty_scores
from src.smooth import FrameSmoother


def _obs(name, peace=0) -> Observation:
    return Observation(has_face=name is not None, name=name, score=0.8 if name else 0.1, expression=empty_scores(), peace_count=peace)


class SmoothTest(unittest.TestCase):
    def test_flicker_does_not_confirm_identity(self):
        smoother = FrameSmoother(5)
        result = None
        for name in (None, "甲", None, "甲", None):
            result = smoother.update(_obs(name))
        self.assertIsNone(result.name)
        self.assertFalse(result.has_face)

    def test_stable_name_and_gesture_mode(self):
        smoother = FrameSmoother(5)
        result = None
        for peace in (1, 1, 1, 2, 0):
            result = smoother.update(_obs("甲", peace))
        self.assertEqual(result.name, "甲")
        self.assertEqual(result.peace_count, 1)


if __name__ == "__main__":
    unittest.main()
