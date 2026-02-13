"""
人脸识别模块 - 综合人脸检测、特征提取和身份识别
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from src.face_detector import FaceDetector
from src.feature_extractor import FeatureExtractor
from src.utils import load_features, save_features, load_image
from config import FACE_RECOGNITION_CONFIG, DATABASE_CONFIG

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """人脸识别器类"""
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化人脸识别器
        
        Args:
            device: 设备类型 ('cpu' 或 'cuda')
        """
        self.device = device
        self.config = FACE_RECOGNITION_CONFIG
        self.db_config = DATABASE_CONFIG
        
        # 初始化组件
        self.detector = FaceDetector(device)
        self.extractor = FeatureExtractor(device)
        
        # 加载特征数据库
        self.database_features = self.load_database()
        
        logger.info("人脸识别器初始化完成")
    
    def load_database(self) -> Dict[str, np.ndarray]:
        """
        加载特征数据库
        
        Returns:
            Dict[str, np.ndarray]: 特征数据库
        """
        features_file = self.db_config['features_file']
        data = load_features(features_file)
        
        if data is None:
            logger.warning("特征数据库为空，请先运行 build_database.py 构建数据库")
            return {}
        
        # 处理新的数据格式（包含features, info, metadata）
        if isinstance(data, dict) and 'features' in data:
            features_dict = data['features']
            logger.info(f"加载特征数据库成功，包含 {len(features_dict)} 个身份")
            return features_dict
        else:
            # 兼容旧格式
            logger.info(f"加载特征数据库成功，包含 {len(data)} 个身份")
            return data
    
    def reload_database(self) -> bool:
        """
        重新加载特征数据库
        
        Returns:
            bool: 是否成功重新加载
        """
        try:
            self.database_features = self.load_database()
            return True
        except Exception as e:
            logger.error(f"重新加载数据库失败: {str(e)}")
            return False
    
    def recognize_face(self, image: np.ndarray, 
                      return_all_faces: bool = False) -> List[Dict[str, Any]]:
        """
        识别图像中的人脸
        
        Args:
            image: 输入图像
            return_all_faces: 是否返回所有检测到的人脸（包括未识别的）
            
        Returns:
            List[Dict]: 识别结果列表，每个元素包含:
                - 'name': 识别的姓名 (如果未识别则为 'Unknown')
                - 'confidence': 置信度
                - 'box': 边界框 [x, y, w, h]
                - 'face_image': 人脸图像
                - 'detection_confidence': 检测置信度
        """
        results = []
        
        try:
            # 检测并提取人脸
            start_time = time.time()
            detected_faces = self.detector.detect_and_extract_faces(image)
            detection_time = time.time() - start_time
            
            if not detected_faces:
                logger.debug("未检测到人脸")
                return results
            
            logger.debug(f"检测到 {len(detected_faces)} 张人脸，耗时 {detection_time:.3f}s")
            
            # 识别每张人脸
            for face_info in detected_faces:
                if not self.detector.is_valid_face(face_info):
                    if return_all_faces:
                        result = {
                            'name': 'Unknown',
                            'confidence': 0.0,
                            'box': face_info['box'],
                            'face_image': face_info['face'],
                            'detection_confidence': face_info['confidence']
                        }
                        results.append(result)
                    continue
                
                # 提取特征
                start_time = time.time()
                features = self.extractor.extract_features(face_info['tensor'])
                extraction_time = time.time() - start_time
                
                if features is None:
                    if return_all_faces:
                        result = {
                            'name': 'Unknown',
                            'confidence': 0.0,
                            'box': face_info['box'],
                            'face_image': face_info['face'],
                            'detection_confidence': face_info['confidence']
                        }
                        results.append(result)
                    continue
                
                # 在数据库中查找匹配
                start_time = time.time()
                best_match, similarity = self.extractor.find_best_match(
                    features, self.database_features
                )
                matching_time = time.time() - start_time
                
                # 构建结果
                result = {
                    'name': best_match or 'Unknown',
                    'confidence': similarity,
                    'box': face_info['box'],
                    'face_image': face_info['face'],
                    'detection_confidence': face_info['confidence'],
                    'processing_time': {
                        'extraction': extraction_time,
                        'matching': matching_time
                    }
                }
                
                # 只有识别成功或要求返回所有人脸时才添加结果
                if best_match is not None or return_all_faces:
                    results.append(result)
                
                logger.debug(f"识别结果: {result['name']} (置信度: {similarity:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"人脸识别失败: {str(e)}")
            return []
    
    def recognize_image_file(self, image_path: str, 
                           return_all_faces: bool = False) -> List[Dict[str, Any]]:
        """
        识别图像文件中的人脸
        
        Args:
            image_path: 图像文件路径
            return_all_faces: 是否返回所有检测到的人脸
            
        Returns:
            List[Dict]: 识别结果列表
        """
        # 加载图像
        image = load_image(image_path)
        if image is None:
            logger.error(f"无法加载图像: {image_path}")
            return []
        
        # 识别人脸
        results = self.recognize_face(image, return_all_faces)
        
        # 添加文件路径信息
        for result in results:
            result['source_file'] = image_path
        
        return results
    
    def batch_recognize(self, image_paths: List[str], 
                       return_all_faces: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量识别多个图像文件
        
        Args:
            image_paths: 图像文件路径列表
            return_all_faces: 是否返回所有检测到的人脸
            
        Returns:
            Dict[str, List[Dict]]: 每个文件的识别结果
        """
        results = {}
        total_files = len(image_paths)
        
        logger.info(f"开始批量识别 {total_files} 个图像文件")
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"处理第 {i}/{total_files} 个文件: {os.path.basename(image_path)}")
            
            try:
                file_results = self.recognize_image_file(image_path, return_all_faces)
                results[image_path] = file_results
                
                # 打印进度
                if len(file_results) > 0:
                    names = [r['name'] for r in file_results]
                    logger.info(f"  检测到: {names}")
                else:
                    logger.info("  未检测到人脸")
                    
            except Exception as e:
                logger.error(f"处理文件 {image_path} 时出错: {str(e)}")
                results[image_path] = []
        
        logger.info(f"批量识别完成，共处理 {total_files} 个文件")
        return results
    
    def add_person_to_database(self, name: str, face_features: np.ndarray) -> bool:
        """
        向数据库添加新的人脸特征
        
        Args:
            name: 人名
            face_features: 人脸特征向量
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 添加到内存中的数据库
            self.database_features[name] = face_features
            
            # 保存到文件
            success = save_features(self.database_features, self.db_config['features_file'])
            
            if success:
                logger.info(f"成功添加新人员: {name}")
                return True
            else:
                # 如果保存失败，从内存中移除
                del self.database_features[name]
                return False
                
        except Exception as e:
            logger.error(f"添加人员到数据库失败: {str(e)}")
            return False
    
    def remove_person_from_database(self, name: str) -> bool:
        """
        从数据库中移除人员
        
        Args:
            name: 人名
            
        Returns:
            bool: 是否移除成功
        """
        try:
            if name not in self.database_features:
                logger.warning(f"数据库中不存在人员: {name}")
                return False
            
            # 从内存中移除
            del self.database_features[name]
            
            # 保存到文件
            success = save_features(self.database_features, self.db_config['features_file'])
            
            if success:
                logger.info(f"成功移除人员: {name}")
                return True
            else:
                logger.error(f"保存数据库失败，无法移除人员: {name}")
                return False
                
        except Exception as e:
            logger.error(f"移除人员失败: {str(e)}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        获取数据库信息
        
        Returns:
            Dict: 数据库信息
        """
        return {
            'total_persons': len(self.database_features),
            'persons': list(self.database_features.keys()),
            'features_file': self.db_config['features_file'],
            'file_exists': os.path.exists(self.db_config['features_file'])
        }
    
    def get_recognition_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取识别统计信息
        
        Args:
            results: 识别结果列表
            
        Returns:
            Dict: 统计信息
        """
        if not results:
            return {'total_faces': 0, 'recognized_faces': 0, 'unknown_faces': 0}
        
        total_faces = len(results)
        recognized_faces = len([r for r in results if r['name'] != 'Unknown'])
        unknown_faces = total_faces - recognized_faces
        
        # 计算平均置信度
        confidences = [r['confidence'] for r in results if r['confidence'] > 0]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # 识别的人员统计
        recognized_names = [r['name'] for r in results if r['name'] != 'Unknown']
        unique_persons = len(set(recognized_names))
        
        return {
            'total_faces': total_faces,
            'recognized_faces': recognized_faces,
            'unknown_faces': unknown_faces,
            'recognition_rate': recognized_faces / total_faces if total_faces > 0 else 0.0,
            'average_confidence': avg_confidence,
            'unique_persons': unique_persons,
            'recognized_names': recognized_names
        }
