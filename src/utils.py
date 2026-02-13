"""
工具函数模块 - 提供通用功能函数
"""

import os
import pickle
import logging
import numpy as np
from PIL import Image
import torch
from typing import List, Tuple, Optional, Dict, Any
import cv2

from config import LOGGING_CONFIG, IMAGE_CONFIG

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format']
)
logger = logging.getLogger(__name__)

def setup_directories(dirs: List[str]) -> None:
    """创建必要的目录"""
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"创建目录: {dir_path}")

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        numpy.ndarray: 图像数组 (BGR格式) 或 None
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return None
            
        # 使用OpenCV加载图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法加载图像: {image_path}")
            return None
            
        return image
    except Exception as e:
        logger.error(f"加载图像时出错 {image_path}: {str(e)}")
        return None

def save_image(image: np.ndarray, save_path: str) -> bool:
    """
    保存图像文件
    
    Args:
        image: 图像数组
        save_path: 保存路径
        
    Returns:
        bool: 是否保存成功
    """
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图像
        success = cv2.imwrite(save_path, image)
        if success:
            logger.info(f"图像已保存: {save_path}")
            return True
        else:
            logger.error(f"保存图像失败: {save_path}")
            return False
    except Exception as e:
        logger.error(f"保存图像时出错 {save_path}: {str(e)}")
        return False

def resize_image(image: np.ndarray, max_size: Tuple[int, int] = None) -> np.ndarray:
    """
    调整图像尺寸
    
    Args:
        image: 输入图像
        max_size: 最大尺寸 (width, height)
        
    Returns:
        numpy.ndarray: 调整后的图像
    """
    if max_size is None:
        max_size = IMAGE_CONFIG['max_image_size']
        
    height, width = image.shape[:2]
    max_width, max_height = max_size
    
    if width <= max_width and height <= max_height:
        return image
        
    # 计算缩放比例
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 调整尺寸
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    logger.debug(f"图像尺寸调整: ({width}, {height}) -> ({new_width}, {new_height})")
    
    return resized

def get_image_files(directory: str) -> List[str]:
    """
    获取目录中所有图像文件
    
    Args:
        directory: 目录路径
        
    Returns:
        List[str]: 图像文件路径列表
    """
    image_files = []
    extensions = IMAGE_CONFIG['image_extensions']
    
    if not os.path.exists(directory):
        logger.warning(f"目录不存在: {directory}")
        return image_files
        
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
                
    logger.info(f"在 {directory} 中找到 {len(image_files)} 个图像文件")
    return image_files

def save_features(features_dict: Dict[str, Any], file_path: str) -> bool:
    """
    保存特征向量到文件
    
    Args:
        features_dict: 特征字典
        file_path: 保存路径
        
    Returns:
        bool: 是否保存成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(features_dict, f)
            
        logger.info(f"特征向量已保存: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存特征向量时出错: {str(e)}")
        return False

def load_features(file_path: str) -> Optional[Dict[str, Any]]:
    """
    从文件加载特征向量
    
    Args:
        file_path: 文件路径
        
    Returns:
        Dict[str, Any]: 特征字典 或 None
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"特征文件不存在: {file_path}")
            return None
            
        with open(file_path, 'rb') as f:
            features_dict = pickle.load(f)
            
        logger.info(f"特征向量已加载: {file_path}")
        return features_dict
    except Exception as e:
        logger.error(f"加载特征向量时出错: {str(e)}")
        return None

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算余弦相似度
    
    Args:
        vec1: 向量1
        vec2: 向量2
        
    Returns:
        float: 余弦相似度值
    """
    try:
        # 归一化向量
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # 计算余弦相似度
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    except Exception as e:
        logger.error(f"计算余弦相似度时出错: {str(e)}")
        return 0.0

def calculate_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算欧氏距离
    
    Args:
        vec1: 向量1
        vec2: 向量2
        
    Returns:
        float: 欧氏距离
    """
    try:
        distance = np.linalg.norm(vec1 - vec2)
        return float(distance)
    except Exception as e:
        logger.error(f"计算欧氏距离时出错: {str(e)}")
        return float('inf')

def draw_bounding_box(image: np.ndarray, box: Tuple[int, int, int, int], 
                     label: str = "", confidence: float = 0.0, 
                     color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    在图像上绘制边界框和标签
    
    Args:
        image: 输入图像
        box: 边界框 (x, y, w, h)
        label: 标签文本
        confidence: 置信度
        color: 颜色 (B, G, R)
        
    Returns:
        numpy.ndarray: 绘制后的图像
    """
    x, y, w, h = box
    
    # 绘制边界框
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    # 准备标签文本
    if label and confidence > 0:
        text = f"{label} ({confidence:.1%})"
    elif label:
        text = label
    else:
        text = f"Unknown ({confidence:.1%})" if confidence > 0 else "Unknown"
    
    # 绘制标签背景
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y), color, -1)
    
    # 绘制标签文本
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def check_device() -> str:
    """
    检查可用设备
    
    Returns:
        str: 设备类型 ('cuda' 或 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        logger.info("使用CPU")
        
    return device

def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
