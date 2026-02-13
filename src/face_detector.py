"""
人脸检测模块 - 使用MTCNN进行人脸检测和对齐
"""

import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import logging
from typing import List, Tuple, Optional, Dict, Any

from config import FACE_DETECTION_CONFIG
from src.utils import check_device

logger = logging.getLogger(__name__)

class FaceDetector:
    """人脸检测器类"""
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化人脸检测器
        
        Args:
            device: 设备类型 ('cpu' 或 'cuda')
        """
        self.device = device or check_device()
        self.config = FACE_DETECTION_CONFIG
        
        # 初始化MTCNN (PyTorch版本)
        try:
            # 确定设备
            if self.device == 'cuda' and torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
                self.device = 'cpu'
            
            # 设置模型缓存目录到项目的models文件夹
            from config import MODELS_DIR
            os.environ['TORCH_HOME'] = MODELS_DIR
            
            self.mtcnn = MTCNN(
                min_face_size=self.config['min_face_size'],
                thresholds=self.config['thresholds'],
                factor=self.config['factor'],
                post_process=self.config['post_process'],
                device=device
            )
            logger.info(f"PyTorch MTCNN人脸检测器初始化成功 (设备: {self.device})")
        except Exception as e:
            logger.error(f"初始化MTCNN失败: {str(e)}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            List[Dict]: 检测结果列表，每个元素包含:
                - 'box': 边界框 [x, y, width, height]
                - 'confidence': 置信度
                - 'keypoints': 关键点字典
        """
        try:
            # 转换BGR到RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # 检测人脸 (facenet_pytorch版本)
            boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
            
            if boxes is None:
                logger.debug("未检测到人脸")
                return []
            
            faces = []
            for i in range(len(boxes)):
                box = boxes[i]
                confidence = probs[i]
                landmark = landmarks[i] if landmarks is not None else None
                
                # 转换边界框格式 [x1, y1, x2, y2] -> [x, y, w, h]
                x1, y1, x2, y2 = box
                x = int(max(0, x1))
                y = int(max(0, y1))
                w = int(min(x2 - x1, image.shape[1] - x))
                h = int(min(y2 - y1, image.shape[0] - y))
                
                # 构建关键点字典
                keypoints = {}
                if landmark is not None:
                    keypoints = {
                        'left_eye': (int(landmark[0][0]), int(landmark[0][1])),
                        'right_eye': (int(landmark[1][0]), int(landmark[1][1])),
                        'nose': (int(landmark[2][0]), int(landmark[2][1])),
                        'mouth_left': (int(landmark[3][0]), int(landmark[3][1])),
                        'mouth_right': (int(landmark[4][0]), int(landmark[4][1]))
                    }
                
                face_info = {
                    'box': [x, y, w, h],
                    'confidence': float(confidence),
                    'keypoints': keypoints
                }
                faces.append(face_info)
            
            logger.debug(f"检测到 {len(faces)} 张人脸")
            return faces
            
        except Exception as e:
            logger.error(f"人脸检测失败: {str(e)}")
            return []
    
    def extract_face(self, image: np.ndarray, box: List[int], 
                    margin: int = 20, size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]:
        """
        从图像中提取人脸区域
        
        Args:
            image: 输入图像
            box: 边界框 [x, y, width, height]
            margin: 边距
            size: 输出尺寸
            
        Returns:
            numpy.ndarray: 提取的人脸图像 或 None
        """
        try:
            x, y, w, h = box
            
            # 添加边距
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            # 提取人脸区域
            face = image[y1:y2, x1:x2]
            
            if face.size == 0:
                logger.warning("提取的人脸区域为空")
                return None
            
            # 调整尺寸
            face_resized = cv2.resize(face, size, interpolation=cv2.INTER_AREA)
            
            return face_resized
            
        except Exception as e:
            logger.error(f"提取人脸失败: {str(e)}")
            return None
    
    def align_face(self, image: np.ndarray, keypoints: Dict[str, Tuple[int, int]], 
                   size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]:
        """
        基于关键点对齐人脸
        
        Args:
            image: 输入图像
            keypoints: 人脸关键点
            size: 输出尺寸
            
        Returns:
            numpy.ndarray: 对齐后的人脸图像 或 None
        """
        try:
            # 获取眼部关键点
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            
            # 计算眼部中心点
            eye_center = ((left_eye[0] + right_eye[0]) // 2, 
                         (left_eye[1] + right_eye[1]) // 2)
            
            # 计算旋转角度
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.arctan2(dy, dx) * 180.0 / np.pi
            
            # 获取旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            
            # 旋转图像
            aligned = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            
            # 重新检测人脸边界框（旋转后可能发生变化）
            faces = self.detect_faces(aligned)
            if not faces:
                logger.warning("对齐后未检测到人脸")
                return None
            
            # 提取对齐后的人脸
            face_box = faces[0]['box']
            aligned_face = self.extract_face(aligned, face_box, size=size)
            
            return aligned_face
            
        except Exception as e:
            logger.error(f"人脸对齐失败: {str(e)}")
            return None
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        预处理人脸图像用于特征提取
        
        Args:
            face_image: 人脸图像 (BGR格式)
            
        Returns:
            torch.Tensor: 预处理后的张量
        """
        try:
            # 转换BGR到RGB
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # 调整尺寸到160x160
            rgb_face = cv2.resize(rgb_face, (160, 160))
            
            # 归一化到[0,1]
            normalized = rgb_face.astype(np.float32) / 255.0
            
            # 标准化 (ImageNet统计值)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std
            
            # 转换为CHW格式
            tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
            
            # 移动到设备并添加batch维度
            tensor = tensor.to(self.device).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            logger.error(f"人脸预处理失败: {str(e)}")
            return None
    
    def detect_and_extract_faces(self, image: np.ndarray, 
                                align_faces: bool = True) -> List[Dict[str, Any]]:
        """
        检测并提取所有人脸
        
        Args:
            image: 输入图像
            align_faces: 是否对齐人脸
            
        Returns:
            List[Dict]: 人脸信息列表，每个元素包含:
                - 'face': 人脸图像
                - 'box': 边界框
                - 'confidence': 置信度
                - 'tensor': 预处理后的张量
        """
        results = []
        
        try:
            # 转换BGR到RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 使用MTCNN检测人脸
            boxes, probs, landmarks = self.mtcnn.detect(rgb_image, landmarks=True)
            
            if boxes is None:
                return results
            
            # 处理每个检测到的人脸
            for i in range(len(boxes)):
                box = boxes[i]
                confidence = probs[i]
                
                # 转换边界框格式 [x1, y1, x2, y2] -> [x, y, w, h]
                x1, y1, x2, y2 = box
                x = int(max(0, x1))
                y = int(max(0, y1))
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                # 确保边界框在图像范围内
                if x >= image.shape[1] or y >= image.shape[0] or w <= 0 or h <= 0:
                    continue
                
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                # 提取人脸区域
                face_region = self.extract_face(image, [x, y, w, h])
                if face_region is None:
                    continue
                
                # 预处理人脸
                face_tensor = self.preprocess_face(face_region)
                if face_tensor is None:
                    continue
                
                result = {
                    'face': face_region,
                    'box': [x, y, w, h],
                    'confidence': float(confidence),
                    'tensor': face_tensor
                }
                results.append(result)
        
        except Exception as e:
            logger.error(f"检测并提取人脸失败: {str(e)}")
        
        return results
    
    def is_valid_face(self, face_info: Dict[str, Any], 
                     min_confidence: float = 0.9) -> bool:
        """
        检查人脸是否有效
        
        Args:
            face_info: 人脸信息
            min_confidence: 最小置信度
            
        Returns:
            bool: 是否有效
        """
        return (face_info['confidence'] >= min_confidence and 
                face_info['face'] is not None and
                face_info['tensor'] is not None)
