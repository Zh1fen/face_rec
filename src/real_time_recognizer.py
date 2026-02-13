"""
Real-time face recognition module - Real-time face recognition using camera
"""

import cv2
import numpy as np
import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.face_recognizer import FaceRecognizer
from src.face_tracker import FaceTracker, TrackedFace
from src.utils import draw_bounding_box, save_image
from config import REAL_TIME_CONFIG

logger = logging.getLogger(__name__)

class RealTimeFaceRecognizer:
    """Real-time face recognizer"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize real-time face recognizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or REAL_TIME_CONFIG
        
        # 初始化组件
        self.recognizer = FaceRecognizer()
        self.tracker = FaceTracker(
            max_disappeared=self.config['max_track_frames'],
            max_distance=50
        ) if self.config['track_faces'] else None
        
        # 摄像头相关
        self.camera = None
        self.camera_id = self.config['camera_id']
        self.fps = self.config['fps']
        self.process_fps = self.config['process_fps']
        
        # 处理相关
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.running = False
        
        # 状态管理
        self.paused = False
        self.current_frame = None
        self.current_results = []
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0.0
        
        # 识别历史
        self.recognition_history = []
        self.max_history = 100
        
        logger.info("实时人脸识别器初始化完成")
    
    def initialize_camera(self, camera_id: Optional[int] = None) -> bool:
        """
        初始化摄像头
        
        Args:
            camera_id: 摄像头ID
            
        Returns:
            bool: 是否初始化成功
        """
        if camera_id is not None:
            self.camera_id = camera_id
        
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                logger.error(f"无法打开摄像头 {self.camera_id}")
                return False
            
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['display_size'][0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['display_size'][1])
            
            # 获取实际参数
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"摄像头 {self.camera_id} 初始化成功")
            logger.info(f"分辨率: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            return False
    
    def list_available_cameras(self) -> List[int]:
        """
        List available cameras
        
        Returns:
            List[int]: List of available camera IDs
        """
        available_cameras = []
        
        for i in range(10):  # Check first 10 cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        logger.info(f"Found available cameras: {available_cameras}")
        return available_cameras
    
    def _process_frame_worker(self):
        """处理帧的工作线程"""
        while self.running:
            try:
                # 获取帧
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=1.0)
                    
                    # 识别人脸
                    start_time = time.time()
                    results = self.recognizer.recognize_face(frame, return_all_faces=True)
                    processing_time = time.time() - start_time
                    
                    # 如果启用跟踪，更新跟踪器
                    if self.tracker and results:
                        tracked_faces = self.tracker.update(frame, results)
                        
                        # 使用跟踪结果更新识别结果
                        for i, result in enumerate(results):
                            for tracked_face in tracked_faces:
                                if self._boxes_overlap(result['box'], tracked_face.box):
                                    result['name'] = tracked_face.name
                                    result['confidence'] = tracked_face.get_average_confidence()
                                    result['tracking_id'] = tracked_face.id
                                    break
                    
                    # 添加处理时间信息
                    for result in results:
                        result['processing_time_total'] = processing_time
                    
                    # 将结果放入队列
                    if not self.result_queue.full():
                        self.result_queue.put(results)
                    
                else:
                    time.sleep(0.01)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理帧时出错: {str(e)}")
    
    def _boxes_overlap(self, box1: List[int], box2: Tuple[int, int, int, int]) -> bool:
        """检查两个边界框是否重叠"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    def _should_process_frame(self) -> bool:
        """判断是否应该处理当前帧"""
        if self.process_fps >= self.fps:
            return True
        
        frame_interval = self.fps / self.process_fps
        return self.frame_count % int(frame_interval) == 0
    
    def _update_fps_counter(self):
        """更新FPS计数器"""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.fps_start_time >= 1.0:
            self.actual_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _add_to_history(self, results: List[Dict[str, Any]]):
        """添加识别结果到历史记录"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        for result in results:
            if result['name'] != 'Unknown':
                history_item = {
                    'timestamp': timestamp,
                    'name': result['name'],
                    'confidence': result['confidence']
                }
                
                self.recognition_history.append(history_item)
                
                # 限制历史记录长度
                if len(self.recognition_history) > self.max_history:
                    self.recognition_history.pop(0)
    
    def _draw_interface(self, frame: np.ndarray) -> np.ndarray:
        """绘制用户界面"""
        height, width = frame.shape[:2]
        
        # 绘制人脸识别结果
        for result in self.current_results:
            box = result['box']
            name = result['name']
            confidence = result.get('confidence', 0.0)
            
            # 选择颜色
            if name == 'Unknown':
                color = (0, 0, 255)  # 红色
            else:
                color = (0, 255, 0)  # 绿色
            
            # 绘制边界框和标签
            frame = draw_bounding_box(frame, box, name, confidence, color)
            
            # 如果启用跟踪，显示跟踪ID
            if 'tracking_id' in result:
                x, y, w, h = box
                tracking_text = f"ID: {result['tracking_id']}"
                cv2.putText(frame, tracking_text, (x, y + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 绘制状态信息
        status_y = 30
        line_height = 25
        
        # FPS信息
        fps_text = f"FPS: {self.actual_fps:.1f} | Process FPS: {self.process_fps}"
        cv2.putText(frame, fps_text, (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += line_height
        
        # 识别统计
        total_faces = len(self.current_results)
        recognized_faces = len([r for r in self.current_results if r['name'] != 'Unknown'])
        stats_text = f"Faces: {total_faces} | Recognized: {recognized_faces}"
        cv2.putText(frame, stats_text, (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += line_height
        
        # 跟踪器信息
        if self.tracker:
            tracker_count = self.tracker.get_tracker_count()
            tracker_text = f"Trackers: {tracker_count}"
            cv2.putText(frame, tracker_text, (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            status_y += line_height
        
        # 暂停状态
        if self.paused:
            pause_text = "PAUSED - Press SPACE to continue"
            cv2.putText(frame, pause_text, (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 控制说明
        help_y = height - 120
        help_texts = [
            "[SPACE] Pause/Resume",
            "[S] Save Frame",
            "[R] Reset History", 
            "[C] Switch Camera",
            "[Q] Quit"
        ]
        
        for text in help_texts:
            cv2.putText(frame, text, (width - 200, help_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            help_y += 20
        
        return frame
    
    def start_recognition(self) -> bool:
        """开始实时识别"""
        if not self.initialize_camera():
            return False
        
        self.running = True
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_frame_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("开始实时人脸识别")
        
        try:
            while self.running:
                # 读取帧
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("无法读取摄像头帧")
                    break
                
                self.current_frame = frame.copy()
                self.frame_count += 1
                
                # 更新FPS计数
                self._update_fps_counter()
                
                # 如果未暂停且应该处理此帧，则添加到处理队列
                if not self.paused and self._should_process_frame():
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                
                # 获取最新的识别结果
                try:
                    while not self.result_queue.empty():
                        self.current_results = self.result_queue.get_nowait()
                        self._add_to_history(self.current_results)
                except queue.Empty:
                    pass
                
                # 绘制界面
                display_frame = self._draw_interface(frame)
                
                # 显示图像
                cv2.imshow('Real-time Face Recognition', display_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key_input(key):
                    break
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止识别")
        except Exception as e:
            logger.error(f"实时识别过程中出错: {str(e)}")
        finally:
            self.stop_recognition()
        
        return True
    
    def _handle_key_input(self, key: int) -> bool:
        """
        处理键盘输入
        
        Args:
            key: 按键代码
            
        Returns:
            bool: 是否继续运行
        """
        if key == ord('q') or key == 27:  # Q键或ESC键
            return False
        elif key == ord(' '):  # 空格键
            self.paused = not self.paused
            logger.info(f"Recognition {'paused' if self.paused else 'resumed'}")
        elif key == ord('s'):  # S键
            self._save_current_frame()
        elif key == ord('r'):  # R键
            self._reset_history()
        elif key == ord('c'):  # C键
            self._switch_camera()
        
        return True
    
    def _save_current_frame(self):
        """保存当前帧"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            
            if save_image(self.current_frame, filename):
                logger.info(f"Current frame saved: {filename}")
    
    def _reset_history(self):
        """重置识别历史"""
        self.recognition_history.clear()
        if self.tracker:
            self.tracker.clear_all()
        logger.info("Recognition history reset")
    
    def _switch_camera(self):
        """切换摄像头"""
        available_cameras = self.list_available_cameras()
        if len(available_cameras) > 1:
            current_index = available_cameras.index(self.camera_id)
            next_index = (current_index + 1) % len(available_cameras)
            new_camera_id = available_cameras[next_index]
            
            # 释放当前摄像头
            if self.camera:
                self.camera.release()
            
            # 初始化新摄像头
            if self.initialize_camera(new_camera_id):
                logger.info(f"Switched to camera {new_camera_id}")
            else:
                logger.error(f"Failed to switch camera, restored to camera {self.camera_id}")
                self.initialize_camera()
    
    def stop_recognition(self):
        """停止识别"""
        self.running = False
        
        # 等待处理线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # 释放摄像头
        if self.camera:
            self.camera.release()
        
        # 关闭窗口
        cv2.destroyAllWindows()
        
        logger.info("Real-time recognition stopped")
    
    def get_recognition_history(self) -> List[Dict[str, Any]]:
        """获取识别历史"""
        return self.recognition_history.copy()
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        stats = {
            'fps': self.actual_fps,
            'frame_count': self.frame_count,
            'current_faces': len(self.current_results),
            'recognized_faces': len([r for r in self.current_results if r['name'] != 'Unknown']),
            'history_count': len(self.recognition_history),
            'paused': self.paused
        }
        
        if self.tracker:
            tracker_stats = self.tracker.get_tracker_stats()
            stats.update(tracker_stats)
        
        return stats
