"""
简单的人脸图片下载脚本
下载一些用于测试的人脸图片
"""

import os
import requests
import logging
from typing import Dict, List
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_face_database_with_sample_images():
    """创建包含示例图片的人脸数据库"""
    
    face_database_dir = "face_database"
    
    # 创建目录结构
    people_data = {
        "Person_A": [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=300&h=300&fit=crop&crop=face",
            "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=300&h=300&fit=crop&crop=face"
        ],
        "Person_B": [
            "https://images.unsplash.com/photo-1494790108755-2616b332c1b3?w=300&h=300&fit=crop&crop=face",
            "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=300&h=300&fit=crop&crop=face"
        ],
        "Person_C": [
            "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=300&h=300&fit=crop&crop=face"
        ],
        "Person_D": [
            "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=300&h=300&fit=crop&crop=face"
        ],
        "Person_E": [
            "https://images.unsplash.com/photo-1519345182560-3f2917c472ef?w=300&h=300&fit=crop&crop=face"
        ]
    }
    
    logger.info("创建人脸数据库目录结构...")
    
    for person_name, urls in people_data.items():
        person_dir = os.path.join(face_database_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        logger.info(f"创建 {person_name} 的文件夹...")
        
        for i, url in enumerate(urls, 1):
            try:
                # 设置请求头
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                logger.info(f"  下载图片 {i}...")
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # 保存图片
                filename = f"photo_{i}.jpg"
                filepath = os.path.join(person_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"  ✅ 保存: {filename}")
                
                # 添加延迟避免请求过快
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"  ❌ 下载失败: {str(e)}")
                continue
    
    # 创建说明文件
    create_database_instructions()
    
    logger.info("✅ 示例人脸数据库创建完成!")
    return True

def create_local_test_images():
    """创建本地测试图片的占位符"""
    
    face_database_dir = "face_database"
    
    # 创建一些示例人员文件夹
    sample_people = [
        "张三", "李四", "王五", "赵六", "陈七"
    ]
    
    for person_name in sample_people:
        person_dir = os.path.join(face_database_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # 创建说明文件
        readme_content = f"""这是 {person_name} 的人脸图片文件夹

请在此文件夹中放入 {person_name} 的照片：

要求：
1. 图片格式：jpg, jpeg, png, bmp
2. 人脸清晰可见
3. 正面或接近正面角度
4. 光线充足
5. 无遮挡（墨镜、帽子等）

建议：
- 放入 2-3 张不同角度的照片
- 确保人脸占图片的主要部分
- 避免模糊或过暗的照片

放入照片后删除此说明文件。
"""
        
        readme_path = os.path.join(person_dir, "放入照片后删除此文件.txt")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    logger.info(f"✅ 已创建 {len(sample_people)} 个人员文件夹")
    logger.info("请在每个文件夹中放入对应人员的照片")

def create_database_instructions():
    """创建数据库使用说明"""
    
    readme_content = """# 人脸数据库使用说明

## 目录结构

face_database/
├── Person_A/           # 人员A的照片
│   ├── photo_1.jpg
│   └── photo_2.jpg
├── Person_B/           # 人员B的照片
│   ├── photo_1.jpg
│   └── photo_2.jpg
└── ...

## 使用步骤

1. **添加您自己的照片**
   - 将现有的示例文件夹重命名为真实姓名
   - 或创建新的文件夹并放入照片

2. **照片要求**
   - 格式：jpg, jpeg, png, bmp, tiff
   - 人脸清晰、正面、光线充足
   - 每人建议 2-3 张不同角度的照片

3. **构建数据库**
   ```bash
   python build_database.py
   ```

4. **开始识别**
   ```bash
   # 离线识别
   python main.py --image test.jpg
   
   # 实时识别
   python real_time_main.py
   ```

## 注意事项

- 文件夹名称就是识别时显示的姓名
- 照片质量直接影响识别准确率
- 建议使用高质量的正面照片
"""
    
    readme_path = os.path.join("face_database", "使用说明.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='创建人脸数据库')
    parser.add_argument('--method', choices=['download', 'local'], default='local',
                       help='创建方法：download(下载示例图片) 或 local(创建本地文件夹)')
    
    args = parser.parse_args()
    
    logger.info("人脸数据库创建工具")
    logger.info("=" * 50)
    
    if args.method == 'download':
        logger.info("方法：下载网络示例图片")
        logger.info("注意：需要网络连接，图片来源于 Unsplash")
        
        try:
            success = create_face_database_with_sample_images()
            if success:
                logger.info("\n✅ 示例数据库创建成功！")
                logger.info("下一步：运行 'python build_database.py' 构建特征数据库")
            else:
                logger.error("创建失败")
        except Exception as e:
            logger.error(f"创建过程中出错: {str(e)}")
            logger.info("尝试使用本地方法: python download_faces.py --method local")
    
    else:  # local
        logger.info("方法：创建本地文件夹结构")
        logger.info("您需要手动添加照片到相应文件夹")
        
        create_local_test_images()
        create_database_instructions()
        
        logger.info("\n✅ 本地文件夹结构创建成功！")
        logger.info("📋 下一步操作：")
        logger.info("1. 在 face_database/ 目录中的各个文件夹里放入对应人员的照片")
        logger.info("2. 删除文件夹中的说明文件")
        logger.info("3. 运行 'python build_database.py' 构建特征数据库")

if __name__ == '__main__':
    main()
