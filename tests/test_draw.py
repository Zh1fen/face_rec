import unittest

import numpy as np

from src.decide import Decision
from src.draw import compose
from src.prompt import parse


class _Face:
    def __init__(self):
        self.bbox = (40, 60, 200, 280)
        self.name = "甲"
        self.score = 0.81
        self.subject = True


class _View:
    def __init__(self):
        self.faces = [_Face()]
        self.hands = []


class DrawTest(unittest.TestCase):
    def test_all_prompts_are_clickable(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        prompts = [
            "请微笑",
            "请做出惊讶的表情",
            "请单手比耶",
            "请双手比耶",
            "请微笑并用单手比耶",
            "请比心",
        ]
        decision = Decision(False, [], "中性 0.00", "比耶 0 只，比心 0 只", "不通过")
        canvas, buttons = compose(frame, _View(), decision, prompts, 4, parse(prompts[4]).summary(), ["甲"], 12)
        self.assertEqual(len(buttons), 6)
        height = canvas.shape[0]
        for rect, _index in buttons:
            self.assertGreaterEqual(rect[1], 0)
            self.assertLess(rect[3], height)


if __name__ == "__main__":
    unittest.main()
