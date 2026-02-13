"""
特征提取模块 - 使用FaceNet提取人脸特征向量
"""

import os
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, List
from facenet_pytorch import InceptionResnetV1

from config import FACE_RECOGNITION_CONFIG
from src.utils import check_device

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """人脸特征提取器类"""
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化特征提取器
        
        Args:
            device: 设备类型 ('cpu' 或 'cuda')
        """
        self.device = device or check_device()
        self.config = FACE_RECOGNITION_CONFIG
        
        # 初始化FaceNet模型
        try:
            # 设置模型缓存目录到项目的models文件夹
            from config import MODELS_DIR
            os.environ['TORCH_HOME'] = MODELS_DIR
            
            self.model = InceptionResnetV1(
                pretrained=self.config['model_name'],
                device=self.device
            )
            self.model.eval()
            
            # 确保模型在正确的设备上
            self.model = self.model.to(self.device)
            
            logger.info(f"FaceNet特征提取器初始化成功 (模型: {self.config['model_name']}, 设备: {self.device})")
            
        except Exception as e:
            logger.error(f"初始化FaceNet失败: {str(e)}")
            raise
    
    def extract_features(self, face_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        从人脸张量提取特征向量
        
        Args:
            face_tensor: 预处理后的人脸张量 [1, 3, H, W]
            
        Returns:
            numpy.ndarray: 特征向量 [512] 或 None
        """
        try:
            if face_tensor is None:
                return None
                
            # 确保张量数据类型为float32
            face_tensor = face_tensor.float()
            
            # 移动到指定设备
            face_tensor = face_tensor.to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.model(face_tensor)
                
            # 转换为numpy数组
            features_np = features.cpu().numpy().flatten()
            
            # 特征归一化
            features_normalized = features_np / np.linalg.norm(features_np)
            
            return features_normalized
            
        except Exception as e:
            logger.error(f"特征提取失败: {str(e)}")
            return None
    
    def extract_batch_features(self, face_tensors: List[torch.Tensor]) -> List[Optional[np.ndarray]]:
        """
        批量提取特征向量
        
        Args:
            face_tensors: 人脸张量列表
            
        Returns:
            List[numpy.ndarray]: 特征向量列表
        """
        if not face_tensors:
            return []
            
        try:
            # 确保所有张量都是float32类型
            face_tensors = [tensor.float() for tensor in face_tensors]
            
            # 将张量堆叠成批次
            batch_tensor = torch.cat(face_tensors, dim=0).to(self.device)
            
            # 批量提取特征
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
            
            # 转换为numpy数组列表
            features_list = []
            for i in range(batch_features.shape[0]):
                features = batch_features[i].cpu().numpy()
                # 归一化
                features_normalized = features / np.linalg.norm(features)
                features_list.append(features_normalized)
            
            logger.debug(f"批量提取了 {len(features_list)} 个特征向量")
            return features_list
            
        except Exception as e:
            logger.error(f"批量特征提取失败: {str(e)}")
            return [None] * len(face_tensors)
    
    def compare_features(self, features1: np.ndarray, features2: np.ndarray, 
                        method: str = 'cosine') -> float:
        """
        比较两个特征向量的相似度
        
        Args:
            features1: 特征向量1
            features2: 特征向量2
            method: 比较方法 ('cosine' 或 'euclidean')
            
        Returns:
            float: 相似度值
        """
        try:
            if method == 'cosine':
                # 余弦相似度
                similarity = np.dot(features1, features2)
                return float(similarity)
            elif method == 'euclidean':
                # 欧氏距离 (转换为相似度: 距离越小，相似度越高)
                distance = np.linalg.norm(features1 - features2)
                similarity = 1.0 / (1.0 + distance)  # 将距离转换为0-1的相似度
                return float(similarity)
            else:
                logger.error(f"不支持的比较方法: {method}")
                return 0.0
                
        except Exception as e:
            logger.error(f"特征比较失败: {str(e)}")
            return 0.0
    
    def find_best_match(self, query_features: np.ndarray, 
                       database_features: dict, 
                       threshold: Optional[float] = None) -> tuple:
        """
        在特征数据库中查找最佳匹配
        
        Args:
            query_features: 查询特征向量
            database_features: 特征数据库 {name: features}
            threshold: 相似度阈值
            
        Returns:
            tuple: (best_match_name, best_similarity)
        """
        if threshold is None:
            threshold = self.config['similarity_threshold']
            
        best_match = None
        best_similarity = 0.0
        
        try:
            for name, db_features in database_features.items():
                # 确保db_features是numpy数组
                if not isinstance(db_features, np.ndarray):
                    logger.warning(f"跳过 {name}: 特征不是numpy数组，类型为 {type(db_features)}")
                    continue
                
                similarity = self.compare_features(query_features, db_features)
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = name
            
            return best_match, best_similarity
            
        except Exception as e:
            logger.error(f"查找最佳匹配失败: {str(e)}")
            return None, 0.0
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 模型信息
        """
        return {
            'model_name': self.config['model_name'],
            'embedding_size': self.config['embedding_size'],
            'device': self.device,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
