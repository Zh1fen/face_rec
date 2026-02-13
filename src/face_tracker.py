"""
人脸跟踪模块 - 用于实时识别中跟踪人脸，避免重复识别
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrackedFace:
    """跟踪的人脸信息"""
    id: int
    name: str
    confidence: float
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    tracker: Any  # OpenCV tracker object
    last_update: float
    recognition_count: int = 1
    total_confidence: float = 0.0
    
    def __post_init__(self):
        self.total_confidence = self.confidence
    
    def update_recognition(self, name: str, confidence: float):
        """更新识别结果"""
        if name == self.name:
            self.recognition_count += 1
            self.total_confidence += confidence
            self.confidence = self.total_confidence / self.recognition_count
        else:
            # 如果识别结果不同，重置统计
            self.name = name
            self.confidence = confidence
            self.recognition_count = 1
            self.total_confidence = confidence
        
        self.last_update = time.time()
    
    def get_average_confidence(self) -> float:
        """获取平均置信度"""
        return self.total_confidence / max(1, self.recognition_count)

class FaceTracker:
    """人脸跟踪器"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: int = 50):
        """
        初始化人脸跟踪器
        
        Args:
            max_disappeared: 最大消失帧数
            max_distance: 最大跟踪距离
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        self.tracked_faces: Dict[int, TrackedFace] = {}
        
        logger.info("人脸跟踪器初始化完成")
    
    def _create_tracker(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> Any:
        """
        创建OpenCV跟踪器
        
        Args:
            image: 当前帧图像
            box: 边界框 (x, y, w, h)
            
        Returns:
            tracker: 跟踪器对象
        """
        try:
            # 使用CSRT跟踪器（精度较高）
            tracker = cv2.TrackerCSRT_create()
            success = tracker.init(image, box)
            
            if success:
                return tracker
            else:
                logger.warning("跟踪器初始化失败")
                return None
                
        except Exception as e:
            logger.error(f"创建跟踪器失败: {str(e)}")
            return None
    
    def _calculate_distance(self, box1: Tuple[int, int, int, int], 
                          box2: Tuple[int, int, int, int]) -> float:
        """
        计算两个边界框中心点的距离
        
        Args:
            box1: 边界框1 (x, y, w, h)
            box2: 边界框2 (x, y, w, h)
            
        Returns:
            float: 距离
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance
    
    def _box_overlap(self, box1: Tuple[int, int, int, int], 
                    box2: Tuple[int, int, int, int]) -> float:
        """
        计算两个边界框的重叠度
        
        Args:
            box1: 边界框1 (x, y, w, h)
            box2: 边界框2 (x, y, w, h)
            
        Returns:
            float: 重叠度 (0-1)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def update(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> List[TrackedFace]:
        """
        更新跟踪器
        
        Args:
            image: 当前帧图像
            detections: 检测结果列表
            
        Returns:
            List[TrackedFace]: 当前跟踪的人脸列表
        """
        current_time = time.time()
        
        # 1. 更新现有跟踪器
        updated_faces = {}
        for face_id, tracked_face in self.tracked_faces.items():
            try:
                success, box = tracked_face.tracker.update(image)
                if success:
                    # 转换box格式 (x, y, w, h)
                    x, y, w, h = map(int, box)
                    tracked_face.box = (x, y, w, h)
                    updated_faces[face_id] = tracked_face
                else:
                    logger.debug(f"跟踪器 {face_id} 更新失败")
            except Exception as e:
                logger.error(f"更新跟踪器 {face_id} 时出错: {str(e)}")
        
        # 2. 匹配检测结果与现有跟踪
        used_detections = set()
        used_trackers = set()
        
        for detection in detections:
            detection_box = tuple(detection['box'])
            best_match_id = None
            best_overlap = 0.0
            
            # 查找最佳匹配的跟踪器
            for face_id, tracked_face in updated_faces.items():
                if face_id in used_trackers:
                    continue
                
                # 计算重叠度
                overlap = self._box_overlap(detection_box, tracked_face.box)
                
                if overlap > best_overlap and overlap > 0.3:  # 重叠度阈值
                    best_overlap = overlap
                    best_match_id = face_id
            
            # 如果找到匹配的跟踪器，更新它
            if best_match_id is not None:
                tracked_face = updated_faces[best_match_id]
                tracked_face.update_recognition(
                    detection.get('name', 'Unknown'),
                    detection.get('confidence', 0.0)
                )
                tracked_face.box = detection_box
                
                used_detections.add(id(detection))
                used_trackers.add(best_match_id)
        
        # 3. 为未匹配的检测创建新跟踪器
        for detection in detections:
            if id(detection) not in used_detections:
                box = tuple(detection['box'])
                tracker = self._create_tracker(image, box)
                
                if tracker is not None:
                    tracked_face = TrackedFace(
                        id=self.next_id,
                        name=detection.get('name', 'Unknown'),
                        confidence=detection.get('confidence', 0.0),
                        box=box,
                        tracker=tracker,
                        last_update=current_time
                    )
                    
                    updated_faces[self.next_id] = tracked_face
                    logger.debug(f"创建新跟踪器 {self.next_id}")
                    self.next_id += 1
        
        # 4. 移除过期的跟踪器
        max_age = self.max_disappeared / 30.0  # 转换为秒
        final_faces = {}
        
        for face_id, tracked_face in updated_faces.items():
            age = current_time - tracked_face.last_update
            if age <= max_age:
                final_faces[face_id] = tracked_face
            else:
                logger.debug(f"移除过期跟踪器 {face_id} (年龄: {age:.1f}s)")
        
        self.tracked_faces = final_faces
        return list(self.tracked_faces.values())
    
    def get_stable_faces(self, min_recognitions: int = 3) -> List[TrackedFace]:
        """
        获取稳定识别的人脸（识别次数达到阈值）
        
        Args:
            min_recognitions: 最小识别次数
            
        Returns:
            List[TrackedFace]: 稳定识别的人脸列表
        """
        stable_faces = []
        for tracked_face in self.tracked_faces.values():
            if (tracked_face.recognition_count >= min_recognitions and 
                tracked_face.name != 'Unknown'):
                stable_faces.append(tracked_face)
        
        return stable_faces
    
    def clear_all(self):
        """清除所有跟踪器"""
        self.tracked_faces.clear()
        self.next_id = 0
        logger.info("清除所有跟踪器")
    
    def get_tracker_count(self) -> int:
        """获取当前跟踪器数量"""
        return len(self.tracked_faces)
    
    def get_tracker_stats(self) -> Dict[str, Any]:
        """
        获取跟踪器统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self.tracked_faces:
            return {
                'total_trackers': 0,
                'recognized_trackers': 0,
                'unknown_trackers': 0,
                'average_recognitions': 0.0
            }
        
        total = len(self.tracked_faces)
        recognized = len([f for f in self.tracked_faces.values() if f.name != 'Unknown'])
        unknown = total - recognized
        
        avg_recognitions = np.mean([f.recognition_count for f in self.tracked_faces.values()])
        
        return {
            'total_trackers': total,
            'recognized_trackers': recognized,
            'unknown_trackers': unknown,
            'average_recognitions': avg_recognitions,
            'tracker_ids': list(self.tracked_faces.keys())
        }
