"""
配置文件 - 人脸识别系统参数配置
"""

import os

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FACE_DATABASE_DIR = os.path.join(PROJECT_ROOT, 'face_database')
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features')
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'test_images')

# 人脸检测配置
FACE_DETECTION_CONFIG = {
    'min_face_size': 20,           # 最小人脸尺寸
    'thresholds': [0.6, 0.7, 0.7], # MTCNN三个网络的阈值
    'factor': 0.709,               # 图像金字塔缩放因子
    'post_process': True,          # 是否后处理
    'device': 'cpu'                # 设备类型 'cpu' 或 'cuda'
}

# 人脸识别配置
FACE_RECOGNITION_CONFIG = {
    'model_name': 'vggface2',      # 预训练模型名称
    'embedding_size': 512,         # 特征向量维度
    'similarity_threshold': 0.6,   # 相似度阈值
    'device': 'cpu'                # 设备类型
}

# 实时识别配置
REAL_TIME_CONFIG = {
    'camera_id': 0,                    # 摄像头ID
    'fps': 30,                         # 视频帧率
    'process_fps': 20,                 # 处理帧率
    'detection_confidence': 0.9,       # 人脸检测置信度
    'recognition_threshold': 0.6,      # 识别阈值
    'face_size': (160, 160),          # 人脸尺寸
    'display_size': (800, 600),        # 显示窗口大小
    'save_unknown_faces': True,        # 是否保存未知人脸
    'show_confidence': True,           # 是否显示置信度
    'track_faces': True,               # 是否启用人脸跟踪
    'max_track_frames': 30,            # 最大跟踪帧数
}

# 数据库配置
DATABASE_CONFIG = {
    'features_file': os.path.join(FEATURES_DIR, 'face_features.pkl'),
    'unknown_faces_dir': os.path.join(PROJECT_ROOT, 'unknown_faces'),
    'backup_features': True,           # 是否备份特征文件
}

# 图像处理配置
IMAGE_CONFIG = {
    'face_crop_size': (160, 160),     # 人脸裁剪尺寸
    'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'max_image_size': (1920, 1080),   # 最大图像尺寸
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'face_recognition.log'
}
