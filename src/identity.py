"""YuNet 检测人脸，SFace 提取身份特征。"""

from dataclasses import dataclass

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

from .models import ensure_models


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]
    det_score: float
    embedding: np.ndarray
    area: int


class IdentityEngine:
    def __init__(self):
        paths = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(str(paths["yunet"]), "", (320, 320), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(paths["sface"]), "")

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        found = self.detector.detect(frame)
        rows = found[1] if isinstance(found, tuple) else found
        if rows is None:
            return []
        faces = []
        for row in rows:
            x, y, w, h = [int(round(value)) for value in row[:4]]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + max(w, 0))
            y2 = min(height, y + max(h, 0))
            if x2 <= x1 or y2 <= y1:
                continue
            aligned = self.recognizer.alignCrop(frame, row)
            feature = np.asarray(self.recognizer.feature(aligned), dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(feature))
            if norm < 1e-6:
                continue
            faces.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    det_score=float(row[-1]),
                    embedding=feature / norm,
                    area=(x2 - x1) * (y2 - y1),
                )
            )
        faces.sort(key=lambda item: item.area, reverse=True)
        return faces

