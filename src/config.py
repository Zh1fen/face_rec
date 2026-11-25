"""路径和判定阈值。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
FACE_DB_DIR = ROOT / "face_database"
PROMPTS_PATH = ROOT / "prompts.json"

# OpenCV SFace 官方示例使用的余弦相似度阈值。
IDENTITY_THRESHOLD = 0.363
SMOOTH_WINDOW = 5

EXPRESSION_THRESHOLD = {
    "smile": 0.40,
    "surprise": 0.38,
    "anger": 0.40,
}

EXPRESSION_LABELS = {
    "smile": "微笑",
    "surprise": "惊讶",
    "anger": "生气",
}

GESTURE_LABELS = {
    "peace_one": "单手比耶",
    "peace_two": "双手比耶",
    "finger_heart": "比心",
}

FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
]

MODEL_FILES = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "sface": (
        "face_recognition_sface_2021dec.onnx",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
    "face_landmarker": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ),
    "hand_landmarker": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ),
}
