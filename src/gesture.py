"""用手部 21 个关键点判断比耶和单手比心。"""

import numpy as np

# 指尖、近端关节。距离手腕更远视为伸直。
INDEX = (8, 6)
MIDDLE = (12, 10)
RING = (16, 14)
PINKY = (20, 18)
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9


def classify_hand(landmarks: np.ndarray) -> str | None:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 2):
        return None
    if is_finger_heart(points):
        return "finger_heart"
    if is_peace(points):
        return "peace"
    return None


def count_gestures(hands: list[np.ndarray]) -> tuple[int, int]:
    peace = 0
    heart = 0
    for hand in hands:
        label = classify_hand(hand)
        if label == "peace":
            peace += 1
        elif label == "finger_heart":
            heart += 1
    return peace, heart


def gesture_satisfied(kind: str, peace_count: int, heart_count: int) -> bool:
    if kind == "peace_one":
        return peace_count == 1
    if kind == "peace_two":
        return peace_count >= 2
    if kind == "finger_heart":
        return heart_count >= 1
    return False


def is_peace(points: np.ndarray) -> bool:
    return (
        _extended(points, *INDEX)
        and _extended(points, *MIDDLE)
        and not _extended(points, *RING)
        and not _extended(points, *PINKY)
    )


def is_finger_heart(points: np.ndarray) -> bool:
    scale = _distance(points[0], points[MIDDLE_MCP])
    if scale < 1e-3:
        return False
    pinch = _distance(points[THUMB_TIP], points[INDEX_TIP])
    others_open = (
        _extended(points, *MIDDLE)
        and _extended(points, *RING)
        and _extended(points, *PINKY)
    )
    return pinch < scale * 0.55 and others_open


def _extended(points: np.ndarray, tip: int, pip: int) -> bool:
    wrist = points[0]
    return _distance(points[tip], wrist) > _distance(points[pip], wrist) * 1.15


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))
