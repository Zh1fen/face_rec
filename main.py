"""
离线人脸识别主程序
支持单张图片、批量图片和文件夹识别
"""

import os
import time
import argparse
import logging
from typing import List

from src.face_recognizer import FaceRecognizer
from src.utils import get_image_files, load_image, save_image, draw_bounding_box
from config import TEST_IMAGES_DIR

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def recognize_single_image(recognizer: FaceRecognizer, image_path: str, 
                         save_result: bool = False, show_all: bool = False):
    """
    识别单张图片
    
    Args:
        recognizer: 人脸识别器
        image_path: 图片路径
        save_result: 是否保存结果图片
        show_all: 是否显示所有检测到的人脸
    """
    logger.info(f"正在识别图片: {os.path.basename(image_path)}")
    
    start_time = time.time()
    results = recognizer.recognize_image_file(image_path, return_all_faces=show_all)
    total_time = time.time() - start_time
    
    if not results:
        logger.info("  ❌ 未检测到人脸")
        return
    
    # 显示识别结果
    logger.info(f"  ✅ 检测到 {len(results)} 张人脸 (耗时: {total_time:.3f}s)")
    
    for i, result in enumerate(results, 1):
        name = result['name']
        confidence = result['confidence']
        detection_conf = result['detection_confidence']
        
        if name == 'Unknown':
            logger.info(f"    人脸 {i}: 未知人员 (检测置信度: {detection_conf:.3f})")
        else:
            logger.info(f"    人脸 {i}: {name} (置信度: {confidence:.3f}, "
                       f"检测置信度: {detection_conf:.3f})")
    
    # 保存结果图片
    if save_result:
        image = load_image(image_path)
        if image is not None:
            # 在图片上绘制识别结果
            for result in results:
                box = result['box']
                name = result['name']
                confidence = result['confidence']
                
                color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
                image = draw_bounding_box(image, box, name, confidence, color)
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"result_{base_name}.jpg"
            
            if save_image(image, output_path):
                logger.info(f"  💾 结果已保存: {output_path}")

def recognize_batch_images(recognizer: FaceRecognizer, image_paths: List[str], 
                         save_results: bool = False, show_all: bool = False):
    """
    批量识别图片
    
    Args:
        recognizer: 人脸识别器
        image_paths: 图片路径列表
        save_results: 是否保存结果图片
        show_all: 是否显示所有检测到的人脸
    """
    logger.info(f"开始批量识别 {len(image_paths)} 张图片")
    
    start_time = time.time()
    all_results = recognizer.batch_recognize(image_paths, return_all_faces=show_all)
    total_time = time.time() - start_time
    
    # 统计结果
    total_faces = 0
    total_recognized = 0
    successful_images = 0
    
    for image_path, results in all_results.items():
        if results:
            successful_images += 1
            total_faces += len(results)
            total_recognized += len([r for r in results if r['name'] != 'Unknown'])
        
        # 显示单张图片结果
        image_name = os.path.basename(image_path)
        if results:
            recognized_names = [r['name'] for r in results if r['name'] != 'Unknown']
            if recognized_names:
                logger.info(f"  📷 {image_name}: {', '.join(set(recognized_names))}")
            else:
                logger.info(f"  📷 {image_name}: 检测到人脸但未识别")
        else:
            logger.info(f"  📷 {image_name}: 未检测到人脸")
        
        # 保存结果图片
        if save_results and results:
            image = load_image(image_path)
            if image is not None:
                for result in results:
                    box = result['box']
                    name = result['name']
                    confidence = result['confidence']
                    
                    color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
                    image = draw_bounding_box(image, box, name, confidence, color)
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = f"batch_result_{base_name}.jpg"
                save_image(image, output_path)
    
    # 显示总体统计
    logger.info(f"\n📊 批量识别完成:")
    logger.info(f"  - 总耗时: {total_time:.2f} 秒")
    logger.info(f"  - 平均每张: {total_time/len(image_paths):.3f} 秒")
    logger.info(f"  - 成功识别图片: {successful_images}/{len(image_paths)}")
    logger.info(f"  - 检测到人脸总数: {total_faces}")
    logger.info(f"  - 成功识别人脸: {total_recognized}")
    if total_faces > 0:
        logger.info(f"  - 识别成功率: {total_recognized/total_faces:.1%}")
    
    # 获取识别统计
    all_face_results = []
    for results in all_results.values():
        all_face_results.extend(results)
    
    if all_face_results:
        stats = recognizer.get_recognition_stats(all_face_results)
        logger.info(f"  - 识别到的不同人员: {stats['unique_persons']}")
        if stats['recognized_names']:
            names_count = {}
            for name in stats['recognized_names']:
                names_count[name] = names_count.get(name, 0) + 1
            
            logger.info(f"  - 人员识别统计:")
            for name, count in sorted(names_count.items()):
                logger.info(f"    * {name}: {count} 次")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='离线人脸识别')
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', type=str, 
                           help='单张图片路径')
    input_group.add_argument('--folder', '-f', type=str,
                           help='图片文件夹路径')
    input_group.add_argument('--batch', '-b', nargs='+',
                           help='多张图片路径')
    
    # 输出选项
    parser.add_argument('--save', '-s', action='store_true',
                       help='保存识别结果图片')
    parser.add_argument('--show-all', action='store_true',
                       help='显示所有检测到的人脸（包括未识别的）')
    
    # 其他选项
    parser.add_argument('--test', action='store_true',
                       help='使用测试图片文件夹')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 初始化识别器
    try:
        logger.info("正在初始化人脸识别器...")
        recognizer = FaceRecognizer()
        
        # 检查数据库
        db_info = recognizer.get_database_info()
        if db_info['total_persons'] == 0:
            logger.error("人脸数据库为空，请先运行 'python build_database.py' 构建数据库")
            return
        
        logger.info(f"数据库已加载，包含 {db_info['total_persons']} 个人员: "
                   f"{', '.join(db_info['persons'])}")
        
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        return
    
    # 确定要处理的图片
    image_paths = []
    
    if args.test:
        # 使用测试文件夹
        if not os.path.exists(TEST_IMAGES_DIR):
            logger.error(f"测试图片文件夹不存在: {TEST_IMAGES_DIR}")
            return
        image_paths = get_image_files(TEST_IMAGES_DIR)
        
    elif args.image:
        # 单张图片
        if not os.path.exists(args.image):
            logger.error(f"图片文件不存在: {args.image}")
            return
        image_paths = [args.image]
        
    elif args.folder:
        # 文件夹
        if not os.path.exists(args.folder):
            logger.error(f"文件夹不存在: {args.folder}")
            return
        image_paths = get_image_files(args.folder)
        
    elif args.batch:
        # 批量图片
        for img_path in args.batch:
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                logger.warning(f"图片文件不存在，跳过: {img_path}")
    
    if not image_paths:
        logger.error("没有找到要处理的图片文件")
        return
    
    logger.info(f"找到 {len(image_paths)} 张图片")
    
    # 开始识别
    if len(image_paths) == 1:
        # 单张图片
        recognize_single_image(
            recognizer, image_paths[0], 
            save_result=args.save, 
            show_all=args.show_all
        )
    else:
        # 批量识别
        recognize_batch_images(
            recognizer, image_paths,
            save_results=args.save,
            show_all=args.show_all
        )

if __name__ == '__main__':
    main()
