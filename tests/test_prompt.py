import unittest

from src.prompt import parse


class PromptTest(unittest.TestCase):
    def test_expression_only(self):
        result = parse("请微笑")
        self.assertEqual(result.expression, "smile")
        self.assertIsNone(result.gesture)
        self.assertIsNone(result.error)

    def test_gesture_only(self):
        result = parse("请单手比耶")
        self.assertIsNone(result.expression)
        self.assertEqual(result.gesture, "peace_one")

    def test_both(self):
        result = parse("请微笑并用单手比耶")
        self.assertEqual(result.expression, "smile")
        self.assertEqual(result.gesture, "peace_one")
        self.assertEqual(result.summary(), "微笑 + 单手比耶")

    def test_surprise_and_two_peace(self):
        result = parse("请做出惊讶的表情，同时双手比耶")
        self.assertEqual(result.expression, "surprise")
        self.assertEqual(result.gesture, "peace_two")

    def test_two_hands_does_not_count_as_one(self):
        result = parse("请双手比耶")
        self.assertEqual(result.gesture, "peace_two")

    def test_finger_heart(self):
        self.assertEqual(parse("请比心").gesture, "finger_heart")
        self.assertEqual(parse("请比爱心").gesture, "finger_heart")

    def test_negation_keeps_the_other_condition(self):
        result = parse("不要笑，请比耶")
        self.assertIsNone(result.expression)
        self.assertEqual(result.gesture, "peace_one")

    def test_negated_smile_phrase(self):
        result = parse("请不要微笑，双手比耶")
        self.assertIsNone(result.expression)
        self.assertEqual(result.gesture, "peace_two")

    def test_two_expressions(self):
        result = parse("请微笑并且惊讶")
        self.assertIsNotNone(result.error)

    def test_two_gestures(self):
        result = parse("请比耶并比心")
        self.assertIsNotNone(result.error)

    def test_unrecognized(self):
        result = parse("你好")
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
