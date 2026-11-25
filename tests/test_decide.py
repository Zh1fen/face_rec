import unittest

from src.decide import Observation, decide
from src.expression import dominant_expression, scores_from_map
from src.prompt import parse


def _scores(**values: float) -> dict[str, float]:
    return scores_from_map(values)


class ExpressionTest(unittest.TestCase):
    def test_smile_score(self):
        scores = _scores(mouthSmileLeft=0.8, mouthSmileRight=0.8)
        self.assertAlmostEqual(scores["smile"], 0.8)
        self.assertEqual(dominant_expression(scores)[0], "smile")

    def test_surprise_score(self):
        scores = _scores(jawOpen=1, eyeWideLeft=1, eyeWideRight=1, browInnerUp=1)
        self.assertAlmostEqual(scores["surprise"], 1.0)

    def test_low_scores_are_neutral(self):
        self.assertEqual(dominant_expression(_scores())[0], "neutral")


class DecideTest(unittest.TestCase):
    def _obs(self, **kwargs) -> Observation:
        base = dict(has_face=True, name="张三", score=0.8, expression=_scores(), peace_count=0, heart_count=0)
        base.update(kwargs)
        return Observation(**base)

    def test_expression_only(self):
        decision = decide(self._obs(expression=_scores(mouthSmileLeft=0.9, mouthSmileRight=0.9)), parse("请微笑"))
        self.assertTrue(decision.passed)
        self.assertEqual([item.title for item in decision.checks], ["身份", "表情"])

    def test_gesture_only_ignores_expression(self):
        decision = decide(self._obs(peace_count=1), parse("请单手比耶"))
        self.assertTrue(decision.passed)
        self.assertEqual([item.title for item in decision.checks], ["身份", "手势"])

    def test_both_must_pass(self):
        smile = _scores(mouthSmileLeft=0.9, mouthSmileRight=0.9)
        requirement = parse("请微笑并用单手比耶")
        missing_hand = decide(self._obs(expression=smile, peace_count=0), requirement)
        self.assertFalse(missing_hand.passed)
        both = decide(self._obs(expression=smile, peace_count=1), requirement)
        self.assertTrue(both.passed)

    def test_wrong_person(self):
        decision = decide(self._obs(name=None, score=0.1, peace_count=1), parse("请单手比耶"))
        self.assertFalse(decision.passed)

    def test_bad_prompt(self):
        decision = decide(self._obs(), parse("你好"))
        self.assertFalse(decision.passed)
        self.assertEqual(decision.checks, [])

    def test_empty_gallery(self):
        decision = decide(self._obs(), parse("请微笑"), gallery_empty=True)
        self.assertFalse(decision.checks[0].ok)


if __name__ == "__main__":
    unittest.main()
