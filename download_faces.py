"""
下载人脸图片并构建数据库
从网络下载公开的人脸图片用于测试
"""

import os
import requests
import time
import logging
from typing import List, Dict
from urllib.parse import urlparse
import hashlib

from src.utils import setup_directories, save_image
from config import FACE_DATABASE_DIR

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 预定义的人脸图片URL列表 (使用免费的测试图片)
FACE_IMAGES_DATA = {
    "Tom_Hanks": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Tom_Hanks_TIFF_2019.jpg/256px-Tom_Hanks_TIFF_2019.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Tom_Hanks_2016.jpg/256px-Tom_Hanks_2016.jpg"
    ],
    "Emma_Watson": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Emma_Watson_2013.jpg/256px-Emma_Watson_2013.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Emma_Watson_HeForShe_2014.jpg/256px-Emma_Watson_HeForShe_2014.jpg"
    ],
    "Morgan_Freeman": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Morgan_Freeman_Deauville_2018.jpg/256px-Morgan_Freeman_Deauville_2018.jpg"
    ],
    "Scarlett_Johansson": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Scarlett_Johansson_by_Gage_Skidmore_2_%28cropped%29.jpg/256px-Scarlett_Johansson_by_Gage_Skidmore_2_%28cropped%29.jpg"
    ],
    "Will_Smith": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/TechCrunch_Disrupt_2019_%2848834434641%29_%28cropped%29.jpg/256px-TechCrunch_Disrupt_2019_%2848834434641%29_%28cropped%29.jpg"
    ],
    "Jennifer_Lawrence": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Jennifer_Lawrence_SDCC_2015_X-Men.jpg/256px-Jennifer_Lawrence_SDCC_2015_X-Men.jpg"
    ],
    "Leonardo_DiCaprio": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Leonardo_DiCaprio_2014.jpg/256px-Leonardo_DiCaprio_2014.jpg"
    ],
    "Angelina_Jolie": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Angelina_Jolie_2_June_2014_%28cropped%29.jpg/256px-Angelina_Jolie_2_June_2014_%28cropped%29.jpg"
    ],
    "Brad_Pitt": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Brad_Pitt_2019_by_Glenn_Francis.jpg/256px-Brad_Pitt_2019_by_Glenn_Francis.jpg"
    ],
    "Robert_Downey_Jr": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg/256px-Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg"
    ]
}

# 备选图片源 (使用 ThisPersonDoesNotExist.com 生成的虚拟人脸)
GENERATED_FACES_NAMES = [
    "Virtual_Person_01", "Virtual_Person_02", "Virtual_Person_03", 
    "Virtual_Person_04", "Virtual_Person_05", "Virtual_Person_06",
    "Virtual_Person_07", "Virtual_Person_08", "Virtual_Person_09",
    "Virtual_Person_10"
]

def download_image(url: str, save_path: str, timeout: int = 10) -> bool:
    """
    下载图片
    
    Args:
        url: 图片URL
        save_path: 保存路径
        timeout: 超时时间
        
    Returns:
        bool: 是否下载成功
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.info(f"正在下载: {url}")
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图片
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # 验证文件大小
        if os.path.getsize(save_path) < 1024:  # 小于1KB可能是错误页面
            logger.warning(f"下载的文件太小，可能下载失败: {save_path}")
            os.remove(save_path)
            return False
        
        logger.info(f"下载成功: {os.path.basename(save_path)}")
        return True
        
    except Exception as e:
        logger.error(f"下载失败 {url}: {str(e)}")
        return False

def download_generated_face(person_name: str, save_dir: str) -> bool:
    """
    下载生成的虚拟人脸
    
    Args:
        person_name: 人员名称
        save_dir: 保存目录
        
    Returns:
        bool: 是否下载成功
    """
    try:
        # 使用 ThisPersonDoesNotExist.com API (每次请求都是不同的人脸)
        url = "https://thispersondoesnotexist.com/image"
        
        # 为了确保每次下载的是不同的图片，添加时间戳
        timestamp = str(int(time.time()))
        seed = hashlib.md5(f"{person_name}_{timestamp}".encode()).hexdigest()[:8]
        
        save_path = os.path.join(save_dir, f"{person_name}_{seed}.jpg")
        
        return download_image(url, save_path)
        
    except Exception as e:
        logger.error(f"下载生成人脸失败 {person_name}: {str(e)}")
        return False

def download_face_database(use_real_celebrities: bool = True, 
                          use_generated_faces: bool = True,
                          max_images_per_person: int = 3) -> bool:
    """
    下载人脸数据库
    
    Args:
        use_real_celebrities: 是否使用真实名人照片
        use_generated_faces: 是否使用生成的虚拟人脸
        max_images_per_person: 每人最大图片数
        
    Returns:
        bool: 是否下载成功
    """
    logger.info("开始下载人脸图片数据库...")
    
    # 确保目录存在
    setup_directories([FACE_DATABASE_DIR])
    
    total_success = 0
    total_attempts = 0
    
    # 下载真实名人照片
    if use_real_celebrities:
        logger.info("\n📥 下载真实名人照片...")
        
        for person_name, urls in FACE_IMAGES_DATA.items():
            logger.info(f"\n处理人员: {person_name}")
            
            person_dir = os.path.join(FACE_DATABASE_DIR, person_name)
            setup_directories([person_dir])
            
            success_count = 0
            for i, url in enumerate(urls[:max_images_per_person], 1):
                total_attempts += 1
                
                # 生成文件名
                ext = os.path.splitext(urlparse(url).path)[1] or '.jpg'
                filename = f"photo_{i}{ext}"
                save_path = os.path.join(person_dir, filename)
                
                # 跳过已存在的文件
                if os.path.exists(save_path):
                    logger.info(f"文件已存在，跳过: {filename}")
                    success_count += 1
                    total_success += 1
                    continue
                
                # 下载图片
                if download_image(url, save_path):
                    success_count += 1
                    total_success += 1
                else:
                    # 如果下载失败，尝试删除可能的空文件
                    if os.path.exists(save_path):
                        os.remove(save_path)
                
                # 添加延迟避免过于频繁的请求
                time.sleep(1)
            
            logger.info(f"  {person_name}: {success_count}/{len(urls[:max_images_per_person])} 张图片下载成功")
    
    # 下载生成的虚拟人脸
    if use_generated_faces:
        logger.info("\n🤖 下载生成的虚拟人脸...")
        logger.info("注意: 这些是AI生成的虚拟人脸，不是真实人物")
        
        for person_name in GENERATED_FACES_NAMES:
            logger.info(f"\n处理虚拟人员: {person_name}")
            
            person_dir = os.path.join(FACE_DATABASE_DIR, person_name)
            setup_directories([person_dir])
            
            success_count = 0
            for i in range(max_images_per_person):
                total_attempts += 1
                
                # 检查是否已有足够的图片
                existing_files = [f for f in os.listdir(person_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(existing_files) >= max_images_per_person:
                    logger.info(f"  {person_name} 已有 {len(existing_files)} 张图片，跳过")
                    success_count = len(existing_files)
                    total_success += max_images_per_person
                    break
                
                # 下载虚拟人脸
                if download_generated_face(person_name, person_dir):
                    success_count += 1
                    total_success += 1
                
                # 添加延迟
                time.sleep(2)  # 生成人脸需要更长延迟
            
            logger.info(f"  {person_name}: {success_count}/{max_images_per_person} 张图片下载成功")
    
    # 显示总结
    logger.info(f"\n📊 下载完成统计:")
    logger.info(f"  总尝试下载: {total_attempts} 张图片")
    logger.info(f"  成功下载: {total_success} 张图片")
    logger.info(f"  成功率: {total_success/total_attempts:.1%}" if total_attempts > 0 else "  成功率: 0%")
    
    # 检查每个人员文件夹
    logger.info(f"\n📁 人员文件夹统计:")
    for item in os.listdir(FACE_DATABASE_DIR):
        item_path = os.path.join(FACE_DATABASE_DIR, item)
        if os.path.isdir(item_path):
            image_files = [f for f in os.listdir(item_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            logger.info(f"  {item}: {len(image_files)} 张图片")
    
    return total_success > 0

def clean_database():
    """清理数据库中的无效图片"""
    logger.info("正在清理数据库...")
    
    if not os.path.exists(FACE_DATABASE_DIR):
        logger.warning(f"数据库目录不存在: {FACE_DATABASE_DIR}")
        return
    
    cleaned_count = 0
    
    for person_name in os.listdir(FACE_DATABASE_DIR):
        person_dir = os.path.join(FACE_DATABASE_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        logger.info(f"清理 {person_name} 的图片...")
        
        for filename in os.listdir(person_dir):
            file_path = os.path.join(person_dir, filename)
            
            # 检查文件大小
            if os.path.getsize(file_path) < 1024:  # 小于1KB
                logger.info(f"  删除过小文件: {filename}")
                os.remove(file_path)
                cleaned_count += 1
                continue
            
            # 检查是否为图片文件
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                logger.info(f"  删除非图片文件: {filename}")
                os.remove(file_path)
                cleaned_count += 1
                continue
        
        # 如果文件夹为空，删除文件夹
        if not os.listdir(person_dir):
            logger.info(f"  删除空文件夹: {person_name}")
            os.rmdir(person_dir)
    
    logger.info(f"清理完成，删除了 {cleaned_count} 个无效文件")

def create_sample_person():
    """创建示例人员文件夹"""
    sample_dir = os.path.join(FACE_DATABASE_DIR, "示例_请替换为真实照片")
    setup_directories([sample_dir])
    
    readme_content = """这是一个示例文件夹。

请替换为您自己的照片：

1. 将此文件夹重命名为人员姓名
2. 删除此README文件
3. 放入该人员的2-3张清晰正面照片

照片要求：
- 格式：jpg, jpeg, png, bmp
- 人脸清晰可见
- 正面或接近正面
- 光线充足
- 无遮挡（墨镜、帽子等）
"""
    
    readme_path = os.path.join(sample_dir, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='下载人脸图片数据库')
    parser.add_argument('--no-celebrities', action='store_true',
                       help='不下载名人照片')
    parser.add_argument('--no-generated', action='store_true',
                       help='不下载生成的虚拟人脸')
    parser.add_argument('--max-images', type=int, default=3,
                       help='每人最大图片数 (默认: 3)')
    parser.add_argument('--clean', action='store_true',
                       help='清理数据库中的无效文件')
    parser.add_argument('--create-sample', action='store_true',
                       help='创建示例人员文件夹')
    
    args = parser.parse_args()
    
    if args.clean:
        clean_database()
        return
    
    if args.create_sample:
        create_sample_person()
        logger.info("示例人员文件夹已创建")
        return
    
    # 下载图片
    use_celebrities = not args.no_celebrities
    use_generated = not args.no_generated
    
    if not use_celebrities and not use_generated:
        logger.error("至少要启用一种图片源")
        return
    
    logger.info("下载配置:")
    logger.info(f"  下载名人照片: {'是' if use_celebrities else '否'}")
    logger.info(f"  下载生成人脸: {'是' if use_generated else '否'}")
    logger.info(f"  每人最大图片数: {args.max_images}")
    
    success = download_face_database(
        use_real_celebrities=use_celebrities,
        use_generated_faces=use_generated,
        max_images_per_person=args.max_images
    )
    
    if success:
        logger.info("\n✅ 人脸图片数据库下载完成!")
        logger.info("下一步: 运行 'python build_database.py' 构建特征数据库")
        
        # 自动清理无效文件
        clean_database()
        
        # 创建示例文件夹
        create_sample_person()
        
    else:
        logger.error("下载失败，请检查网络连接或稍后重试")

if __name__ == '__main__':
    main()
