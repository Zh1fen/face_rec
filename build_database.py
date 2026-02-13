"""
构建人脸特征数据库
从 face_database 目录中的图片提取特征向量并保存
"""

import os
import time
import logging
from tqdm import tqdm
from typing import Dict, List

from src.face_detector import FaceDetector
from src.feature_extractor import FeatureExtractor
from src.utils import get_image_files, save_features, load_image, setup_directories
from config import FACE_DATABASE_DIR, FEATURES_DIR, DATABASE_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_features_from_person_folder(person_name: str, person_folder: str, 
                                      detector: FaceDetector, 
                                      extractor: FeatureExtractor) -> List[tuple]:
    """
    从单个人员文件夹中提取特征
    
    Args:
        person_name: 人员姓名
        person_folder: 人员文件夹路径
        detector: 人脸检测器
        extractor: 特征提取器
        
    Returns:
        List[tuple]: (特征向量, 图片路径) 列表
    """
    image_files = get_image_files(person_folder)
    if not image_files:
        logger.warning(f"在 {person_folder} 中没有找到图片文件")
        return []
    
    features_list = []
    
    logger.info(f"处理 {person_name} 的 {len(image_files)} 张图片")
    
    for image_path in tqdm(image_files, desc=f"提取 {person_name} 的特征"):
        try:
            # 加载图片
            image = load_image(image_path)
            if image is None:
                logger.warning(f"无法加载图片: {image_path}")
                continue
            
            # 检测人脸
            detected_faces = detector.detect_and_extract_faces(image)
            
            if not detected_faces:
                logger.warning(f"在 {image_path} 中未检测到人脸")
                continue
            
            # 处理检测到的所有人脸
            for i, face_info in enumerate(detected_faces):
                if not detector.is_valid_face(face_info):
                    continue
                
                # 提取特征
                features = extractor.extract_features(face_info['tensor'])
                if features is not None:
                    features_list.append((features, image_path, i))
                    logger.debug(f"成功提取特征: {image_path} (人脸 {i})")
                else:
                    logger.warning(f"特征提取失败: {image_path} (人脸 {i})")
        
        except Exception as e:
            logger.error(f"处理图片 {image_path} 时出错: {str(e)}")
            continue
    
    logger.info(f"从 {person_name} 的图片中成功提取了 {len(features_list)} 个特征向量")
    return features_list

def calculate_average_features(features_list: List[tuple]) -> tuple:
    """
    计算平均特征向量
    
    Args:
        features_list: 特征向量列表
        
    Returns:
        tuple: (平均特征向量, 特征数量, 来源信息)
    """
    if not features_list:
        return None, 0, []
    
    # 提取所有特征向量
    features_array = [item[0] for item in features_list]
    
    # 计算平均值
    import numpy as np
    average_features = np.mean(features_array, axis=0)
    
    # 归一化
    average_features = average_features / np.linalg.norm(average_features)
    
    # 收集来源信息
    sources = [(item[1], item[2]) for item in features_list]
    
    return average_features, len(features_list), sources

def build_face_database(use_average: bool = True, 
                       min_faces_per_person: int = 1) -> bool:
    """
    构建人脸特征数据库
    
    Args:
        use_average: 是否使用平均特征（如果False，使用第一个检测到的特征）
        min_faces_per_person: 每个人最少需要的人脸数量
        
    Returns:
        bool: 是否构建成功
    """
    logger.info("开始构建人脸特征数据库")
    
    # 检查目录
    if not os.path.exists(FACE_DATABASE_DIR):
        logger.error(f"人脸数据库目录不存在: {FACE_DATABASE_DIR}")
        logger.info("请在 face_database/ 目录中按人名创建子文件夹并放入相应的照片")
        return False
    
    # 确保输出目录存在
    setup_directories([FEATURES_DIR])
    
    # 获取所有人员文件夹
    person_folders = []
    for item in os.listdir(FACE_DATABASE_DIR):
        person_path = os.path.join(FACE_DATABASE_DIR, item)
        if os.path.isdir(person_path):
            person_folders.append((item, person_path))
    
    if not person_folders:
        logger.error(f"在 {FACE_DATABASE_DIR} 中没有找到人员文件夹")
        logger.info("请按以下结构组织文件:")
        logger.info("face_database/")
        logger.info("  ├── 张三/")
        logger.info("  │   ├── photo1.jpg")
        logger.info("  │   └── photo2.jpg")
        logger.info("  └── 李四/")
        logger.info("      └── photo1.jpg")
        return False
    
    logger.info(f"发现 {len(person_folders)} 个人员文件夹: {[name for name, _ in person_folders]}")
    
    # 初始化检测器和提取器
    try:
        logger.info("初始化人脸检测器和特征提取器...")
        detector = FaceDetector()
        extractor = FeatureExtractor()
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        return False
    
    # 存储所有特征
    database_features = {}
    database_info = {}
    
    total_start_time = time.time()
    
    # 处理每个人员文件夹
    for person_name, person_folder in person_folders:
        logger.info(f"\n处理人员: {person_name}")
        
        start_time = time.time()
        features_list = extract_features_from_person_folder(
            person_name, person_folder, detector, extractor
        )
        processing_time = time.time() - start_time
        
        if len(features_list) < min_faces_per_person:
            logger.warning(f"跳过 {person_name}: 检测到的人脸数量({len(features_list)}) "
                          f"少于最小要求({min_faces_per_person})")
            continue
        
        if use_average:
            # 使用平均特征
            avg_features, count, sources = calculate_average_features(features_list)
            if avg_features is not None:
                database_features[person_name] = avg_features
                database_info[person_name] = {
                    'method': 'average',
                    'feature_count': count,
                    'sources': sources,
                    'processing_time': processing_time
                }
                logger.info(f"已保存 {person_name} 的平均特征 (基于 {count} 个特征)")
        else:
            # 使用第一个特征
            if features_list:
                database_features[person_name] = features_list[0][0]
                database_info[person_name] = {
                    'method': 'first',
                    'feature_count': 1,
                    'sources': [(features_list[0][1], features_list[0][2])],
                    'processing_time': processing_time
                }
                logger.info(f"已保存 {person_name} 的特征 (来源: {features_list[0][1]})")
    
    total_processing_time = time.time() - total_start_time
    
    # 检查是否有有效特征
    if not database_features:
        logger.error("没有成功提取任何特征，数据库构建失败")
        return False
    
    # 保存特征数据库
    features_file = DATABASE_CONFIG['features_file']
    
    # 准备保存的数据
    save_data = {
        'features': database_features,
        'info': database_info,
        'metadata': {
            'total_persons': len(database_features),
            'build_time': time.time(),
            'processing_time': total_processing_time,
            'use_average': use_average,
            'min_faces_per_person': min_faces_per_person
        }
    }
    
    if save_features(save_data, features_file):
        logger.info(f"\n✅ 数据库构建成功!")
        logger.info(f"📊 统计信息:")
        logger.info(f"  - 总人数: {len(database_features)}")
        logger.info(f"  - 特征文件: {features_file}")
        logger.info(f"  - 总耗时: {total_processing_time:.2f} 秒")
        logger.info(f"  - 平均每人: {total_processing_time/len(database_features):.2f} 秒")
        
        # 显示每个人的详细信息
        logger.info(f"\n📋 人员详情:")
        for name, info in database_info.items():
            logger.info(f"  - {name}: {info['feature_count']} 个特征, "
                       f"{info['processing_time']:.2f}s")
        
        return True
    else:
        logger.error("保存特征数据库失败")
        return False

def validate_database() -> bool:
    """
    验证数据库完整性
    
    Returns:
        bool: 数据库是否有效
    """
    features_file = DATABASE_CONFIG['features_file']
    
    if not os.path.exists(features_file):
        logger.error(f"特征数据库文件不存在: {features_file}")
        return False
    
    try:
        from src.utils import load_features
        data = load_features(features_file)
        
        if data is None:
            logger.error("无法加载特征数据库")
            return False
        
        # 检查数据结构
        if 'features' not in data:
            logger.error("数据库缺少 'features' 字段")
            return False
        
        features = data['features']
        info = data.get('info', {})
        metadata = data.get('metadata', {})
        
        logger.info(f"✅ 数据库验证成功")
        logger.info(f"  - 人员数量: {len(features)}")
        logger.info(f"  - 构建时间: {time.ctime(metadata.get('build_time', 0))}")
        
        # 验证特征向量
        import numpy as np
        for name, feature_vec in features.items():
            if not isinstance(feature_vec, np.ndarray):
                logger.error(f"人员 {name} 的特征不是numpy数组")
                return False
            
            if feature_vec.shape[0] != 512:  # FaceNet特征维度
                logger.warning(f"人员 {name} 的特征维度异常: {feature_vec.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"验证数据库时出错: {str(e)}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='构建人脸特征数据库')
    parser.add_argument('--no-average', action='store_true', 
                       help='不使用平均特征，只使用第一个检测到的特征')
    parser.add_argument('--min-faces', type=int, default=1,
                       help='每个人最少需要的人脸数量 (默认: 1)')
    parser.add_argument('--validate', action='store_true',
                       help='验证现有数据库')
    
    args = parser.parse_args()
    
    if args.validate:
        # 验证数据库
        if validate_database():
            logger.info("数据库验证通过")
        else:
            logger.error("数据库验证失败")
            exit(1)
    else:
        # 构建数据库
        use_average = not args.no_average
        success = build_face_database(
            use_average=use_average,
            min_faces_per_person=args.min_faces
        )
        
        if success:
            logger.info("数据库构建完成，可以开始使用人脸识别系统")
        else:
            logger.error("数据库构建失败")
            exit(1)

if __name__ == '__main__':
    main()
