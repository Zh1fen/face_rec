"""MediaPipe 读取表情系数和手部关键点。"""

import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .expression import empty_scores, scores_from_categories
from .gesture import classify_hand
from .models import ensure_models


@dataclass
class FaceExpression:
    bbox: tuple[float, float, float, float]
    scores: dict[str, float]


@dataclass
class HandObservation:
    points: np.ndarray
    label: str | None


class VisionEngine:
    def __init__(self):
        paths = ensure_models()
        self._timestamp = 0
        self.faces = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(paths["face_landmarker"])),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=4,
                output_face_blendshapes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
        )
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(paths["hand_landmarker"])),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    def read(self, frame: np.ndarray) -> tuple[list[FaceExpression], list[HandObservation]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp += 33
        height, width = frame.shape[:2]
        face_result = self.faces.detect_for_video(image, self._timestamp)
        hand_result = self.hands.detect_for_video(image, self._timestamp)
        return _faces(face_result, width, height), _hands(hand_result)

    def close(self) -> None:
        self.faces.close()
        self.hands.close()


def _faces(result, width: int, height: int) -> list[FaceExpression]:
    expressions = []
    landmarks = result.face_landmarks or []
    blendshapes = result.face_blendshapes or []
    for index, points in enumerate(landmarks):
        xs = [point.x * width for point in points]
        ys = [point.y * height for point in points]
        categories = blendshapes[index] if index < len(blendshapes) else None
        expressions.append(
            FaceExpression(
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                scores=scores_from_categories(categories) if categories else empty_scores(),
            )
        )
    return expressions


def _hands(result) -> list[HandObservation]:
    observations = []
    for points in result.hand_landmarks or []:
        array = np.array([[point.x, point.y] for point in points], dtype=np.float32)
        observations.append(HandObservation(array, classify_hand(array)))
    return observations


def match_expression(face_bbox: tuple[int, int, int, int], expressions: list[FaceExpression]) -> dict[str, float]:
    best = empty_scores()
    best_iou = 0.1
    for item in expressions:
        overlap = _iou(face_bbox, item.bbox)
        if overlap > best_iou:
            best_iou = overlap
            best = item.scores
    return best


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    if union <= 0:
        return 0.0
    return intersection / union
