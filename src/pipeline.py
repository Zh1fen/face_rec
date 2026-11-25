"""一帧里同时看身份、表情和手势。核验对象是画面中最大的人脸。"""

from dataclasses import dataclass

import numpy as np

from .decide import Observation
from .expression import empty_scores
from .identity import IdentityEngine
from .vision import HandObservation, VisionEngine, match_expression


@dataclass
class FaceView:
    bbox: tuple[int, int, int, int]
    name: str | None
    score: float
    subject: bool


@dataclass
class FrameView:
    observation: Observation
    faces: list[FaceView]
    hands: list[HandObservation]


class Engine:
    def __init__(self, gallery, threshold: float):
        self.gallery = gallery
        self.threshold = threshold
        self.identity = IdentityEngine()
        self.vision = VisionEngine()

    def analyze(self, frame: np.ndarray) -> FrameView:
        detected = self.identity.detect(frame)
        expressions, hands = self.vision.read(frame)
        faces: list[FaceView] = []
        subject_observation = Observation(False, None, 0.0, empty_scores(), _peace(hands), _heart(hands))
        for index, face in enumerate(detected):
            matched = self.gallery.match(face.embedding, self.threshold)
            view = FaceView(face.bbox, matched.name, matched.score, index == 0)
            faces.append(view)
            if index == 0:
                subject_observation = Observation(
                    has_face=True,
                    name=matched.name,
                    score=matched.score,
                    expression=match_expression(face.bbox, expressions),
                    peace_count=_peace(hands),
                    heart_count=_heart(hands),
                )
        if not detected:
            subject_observation.peace_count = _peace(hands)
            subject_observation.heart_count = _heart(hands)
        return FrameView(subject_observation, faces, hands)

    def close(self) -> None:
        self.vision.close()


def _peace(hands: list[HandObservation]) -> int:
    return sum(1 for hand in hands if hand.label == "peace")


def _heart(hands: list[HandObservation]) -> int:
    return sum(1 for hand in hands if hand.label == "finger_heart")
