"""
实时人脸识别主程序
使用摄像头进行实时人脸检测和识别
"""

import argparse
import logging
import json
import os

from src.real_time_recognizer import RealTimeFaceRecognizer
from config import REAL_TIME_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_custom_config(config_file: str) -> dict:
    """
    加载自定义配置文件
     
    Args:
        config_file: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
        
        # 合并默认配置和自定义配置
        merged_config = REAL_TIME_CONFIG.copy()
        merged_config.update(custom_config)
        
        logger.info(f"已加载自定义配置: {config_file}")
        return merged_config
        
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return REAL_TIME_CONFIG

def list_available_cameras():
    """列出可用摄像头"""
    logger.info("正在检测可用摄像头...")
    
    recognizer = RealTimeFaceRecognizer()
    cameras = recognizer.list_available_cameras()
    
    if cameras:
        logger.info("发现以下可用摄像头:")
        for cam_id in cameras:
            logger.info(f"  摄像头 {cam_id}")
    else:
        logger.warning("未发现可用摄像头")
    
    return cameras

def create_sample_config():
    """创建示例配置文件"""
    sample_config = {
        "camera_id": 0,
        "fps": 30,
        "process_fps": 10,
        "detection_confidence": 0.9,
        "recognition_threshold": 0.6,
        "face_size": [160, 160],
        "display_size": [800, 600],
        "save_unknown_faces": True,
        "show_confidence": True,
        "track_faces": True,
        "max_track_frames": 30
    }
    
    config_file = "custom_config.json"
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"示例配置文件已创建: {config_file}")
        logger.info("您可以修改此文件中的参数，然后使用 --config 参数加载")
        
    except Exception as e:
        logger.error(f"创建配置文件失败: {str(e)}")

def test_camera(camera_id: int):
    """测试指定摄像头"""
    logger.info(f"正在测试摄像头 {camera_id}...")
    
    import cv2
    
    try:
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"无法打开摄像头 {camera_id}")
            return False
        
        # 获取摄像头信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"摄像头 {camera_id} 信息:")
        logger.info(f"  分辨率: {width} x {height}")
        logger.info(f"  帧率: {fps}")
        
        # 测试读取帧
        ret, frame = cap.read()
        if ret:
            logger.info("✅ 摄像头测试成功")
            
            # 显示测试画面5秒
            logger.info("显示测试画面5秒，按任意键退出...")
            start_time = cv2.getTickCount()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 在画面上显示信息
                cv2.putText(frame, f"Camera {camera_id} Test", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Resolution: {width}x{height}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press any key to exit", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow(f'Camera {camera_id} Test', frame)
                
                # 检查按键或5秒超时
                key = cv2.waitKey(1) & 0xFF
                current_time = cv2.getTickCount()
                elapsed = (current_time - start_time) / cv2.getTickFrequency()
                
                if key != 255 or elapsed > 5:
                    break
            
            cv2.destroyAllWindows()
        else:
            logger.error("无法读取摄像头帧")
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        logger.error(f"测试摄像头时出错: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时人脸识别')
    
    # 基本选项
    parser.add_argument('--camera', '-c', type=int, default=None,
                       help='指定摄像头ID (默认: 0)')
    parser.add_argument('--config', type=str,
                       help='自定义配置文件路径')
    
    # 功能选项
    parser.add_argument('--list-cameras', action='store_true',
                       help='列出可用摄像头')
    parser.add_argument('--test-camera', type=int, metavar='ID',
                       help='测试指定摄像头')
    parser.add_argument('--create-config', action='store_true',
                       help='创建示例配置文件')
    
    # 识别选项
    parser.add_argument('--no-tracking', action='store_true',
                       help='禁用人脸跟踪')
    parser.add_argument('--process-fps', type=int,
                       help='处理帧率 (默认: 10)')
    parser.add_argument('--threshold', type=float,
                       help='识别阈值 (默认: 0.6)')
    
    # 其他选项
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 处理功能选项
    if args.list_cameras:
        list_available_cameras()
        return
    
    if args.test_camera is not None:
        success = test_camera(args.test_camera)
        if success:
            logger.info("摄像头测试完成")
        else:
            logger.error("摄像头测试失败")
        return
    
    if args.create_config:
        create_sample_config()
        return
    
    # 加载配置
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"配置文件不存在: {args.config}")
            return
        config = load_custom_config(args.config)
    else:
        config = REAL_TIME_CONFIG.copy()
    
    # 应用命令行参数覆盖配置
    if args.camera is not None:
        config['camera_id'] = args.camera
    if args.no_tracking:
        config['track_faces'] = False
    if args.process_fps:
        config['process_fps'] = args.process_fps
    if args.threshold:
        config['recognition_threshold'] = args.threshold
    
    # 验证配置
    if config['process_fps'] > config['fps']:
        logger.warning(f"处理帧率({config['process_fps']}) 高于摄像头帧率({config['fps']})，"
                      f"自动调整为 {config['fps']}")
        config['process_fps'] = config['fps']
    
    # 显示配置信息
    logger.info("Real-time recognition configuration:")
    logger.info(f"  Camera ID: {config['camera_id']}")
    logger.info(f"  Video FPS: {config['fps']}")
    logger.info(f"  Process FPS: {config['process_fps']}")
    logger.info(f"  Recognition threshold: {config['recognition_threshold']}")
    logger.info(f"  Face tracking: {'Enabled' if config['track_faces'] else 'Disabled'}")
    logger.info(f"  Display size: {config['display_size'][0]}x{config['display_size'][1]}")
    
    # 检查数据库
    try:
        from src.face_recognizer import FaceRecognizer
        temp_recognizer = FaceRecognizer()
        db_info = temp_recognizer.get_database_info()
        
        if db_info['total_persons'] == 0:
            logger.error("人脸数据库为空，请先运行 'python build_database.py' 构建数据库")
            return
        
        logger.info(f"数据库已加载，包含 {db_info['total_persons']} 个人员")
        
    except Exception as e:
        logger.error(f"检查数据库失败: {str(e)}")
        return
    
    # 检查摄像头可用性
    available_cameras = list_available_cameras()
    if config['camera_id'] not in available_cameras:
        logger.error(f"摄像头 {config['camera_id']} 不可用")
        if available_cameras:
            logger.info(f"建议使用: {available_cameras}")
        return
    
    # 启动实时识别
    try:
        logger.info("\n🎥 Starting Real-time Face Recognition...")
        logger.info("Control Instructions:")
        logger.info("  [SPACE] - Pause/Resume recognition")
        logger.info("  [S] - Save current frame")
        logger.info("  [R] - Reset recognition history")
        logger.info("  [C] - Switch camera")
        logger.info("  [Q] - Quit program")
        logger.info("")
        
        recognizer = RealTimeFaceRecognizer(config)
        success = recognizer.start_recognition()
        
        if success:
            logger.info("Real-time recognition ended normally")
        else:
            logger.error("Real-time recognition ended with error")
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping...")
    except Exception as e:
        logger.error(f"Error in real-time recognition: {str(e)}")
    
    logger.info("Program exited")

if __name__ == '__main__':
    main()
