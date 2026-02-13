"""
将PyTorch模型转换为ONNX格式
用于边缘设备部署和推理加速
"""

import os
import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from config import MODELS_DIR, FACE_DETECTION_CONFIG
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_facenet_to_onnx():
    """转换FaceNet模型为ONNX格式"""
    
    try:
        # 设置输出目录
        onnx_dir = os.path.join(MODELS_DIR, 'onnx')
        os.makedirs(onnx_dir, exist_ok=True)
        
        # 初始化FaceNet模型
        logger.info("加载FaceNet模型...")
        model = InceptionResnetV1(pretrained='vggface2')
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(1, 3, 160, 160)
        
        # 转换为ONNX
        onnx_path = os.path.join(onnx_dir, 'facenet_vggface2.onnx')
        logger.info(f"开始转换FaceNet模型为ONNX格式...")
        
        torch.onnx.export(
            model,                          # 模型
            dummy_input,                    # 示例输入
            onnx_path,                      # 输出路径
            export_params=True,             # 导出参数
            opset_version=11,               # ONNX opset版本
            do_constant_folding=True,       # 常量折叠优化
            input_names=['input'],          # 输入名称
            output_names=['output'],        # 输出名称
            dynamic_axes={                  # 动态轴
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # 验证ONNX模型
        logger.info("验证ONNX模型...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # 测试ONNX推理
        logger.info("测试ONNX推理...")
        ort_session = onnxruntime.InferenceSession(onnx_path)
        
        # 创建测试输入
        test_input = np.random.randn(1, 3, 160, 160).astype(np.float32)
        
        # PyTorch推理
        with torch.no_grad():
            pytorch_output = model(torch.from_numpy(test_input)).numpy()
        
        # ONNX推理
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # 比较输出
        diff = np.abs(pytorch_output - onnx_output).max()
        logger.info(f"PyTorch vs ONNX 最大差异: {diff}")
        
        if diff < 1e-5:
            logger.info("✅ FaceNet模型转换成功！")
        else:
            logger.warning(f"⚠️ 转换后精度有差异，最大差异: {diff}")
        
        return onnx_path
        
    except Exception as e:
        logger.error(f"FaceNet模型转换失败: {str(e)}")
        return None

def convert_mtcnn_to_onnx():
    """转换MTCNN模型为ONNX格式"""
    
    try:
        # 设置输出目录
        onnx_dir = os.path.join(MODELS_DIR, 'onnx')
        os.makedirs(onnx_dir, exist_ok=True)
        
        logger.info("开始转换MTCNN模型...")
        
        # 初始化MTCNN
        mtcnn = MTCNN(
            min_face_size=FACE_DETECTION_CONFIG['min_face_size'],
            thresholds=FACE_DETECTION_CONFIG['thresholds'],
            factor=FACE_DETECTION_CONFIG['factor']
        )
        
        # 转换PNet
        logger.info("转换PNet...")
        dummy_input_pnet = torch.randn(1, 3, 12, 12)
        pnet_path = os.path.join(onnx_dir, 'mtcnn_pnet.onnx')
        
        torch.onnx.export(
            mtcnn.pnet,
            dummy_input_pnet,
            pnet_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['prob', 'bbox'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'prob': {0: 'batch_size'},
                'bbox': {0: 'batch_size'}
            }
        )
        
        # 转换RNet
        logger.info("转换RNet...")
        dummy_input_rnet = torch.randn(1, 3, 24, 24)
        rnet_path = os.path.join(onnx_dir, 'mtcnn_rnet.onnx')
        
        torch.onnx.export(
            mtcnn.rnet,
            dummy_input_rnet,
            rnet_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['prob', 'bbox'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'prob': {0: 'batch_size'},
                'bbox': {0: 'batch_size'}
            }
        )
        
        # 转换ONet
        logger.info("转换ONet...")
        dummy_input_onet = torch.randn(1, 3, 48, 48)
        onet_path = os.path.join(onnx_dir, 'mtcnn_onet.onnx')
        
        torch.onnx.export(
            mtcnn.onet,
            dummy_input_onet,
            onet_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['prob', 'bbox', 'landmarks'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'prob': {0: 'batch_size'},
                'bbox': {0: 'batch_size'},
                'landmarks': {0: 'batch_size'}
            }
        )
        
        # 验证MTCNN模型
        for name, path in [('PNet', pnet_path), ('RNet', rnet_path), ('ONet', onet_path)]:
            logger.info(f"验证{name}模型...")
            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
        
        logger.info("✅ MTCNN模型转换成功！")
        return [pnet_path, rnet_path, onet_path]
        
    except Exception as e:
        logger.error(f"MTCNN模型转换失败: {str(e)}")
        return None

def optimize_onnx_models():
    """优化ONNX模型以提升推理性能"""
    
    try:
        import onnxoptimizer
        
        onnx_dir = os.path.join(MODELS_DIR, 'onnx')
        optimized_dir = os.path.join(onnx_dir, 'optimized')
        os.makedirs(optimized_dir, exist_ok=True)
        
        model_files = [
            'facenet_vggface2.onnx',
            'mtcnn_pnet.onnx',
            'mtcnn_rnet.onnx',
            'mtcnn_onet.onnx'
        ]
        
        for model_file in model_files:
            model_path = os.path.join(onnx_dir, model_file)
            if os.path.exists(model_path):
                logger.info(f"优化模型: {model_file}")
                
                # 加载原始模型
                model = onnx.load(model_path)
                
                # 应用优化
                optimized_model = onnxoptimizer.optimize(model)
                
                # 保存优化后的模型
                optimized_path = os.path.join(optimized_dir, model_file)
                onnx.save(optimized_model, optimized_path)
                
                # 比较模型大小
                original_size = os.path.getsize(model_path) / (1024 * 1024)
                optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
                
                logger.info(f"  原始大小: {original_size:.2f} MB")
                logger.info(f"  优化后大小: {optimized_size:.2f} MB")
                logger.info(f"  压缩比: {(1 - optimized_size/original_size)*100:.1f}%")
        
        logger.info("✅ 模型优化完成！")
        
    except ImportError:
        logger.warning("onnxoptimizer未安装，跳过模型优化")
        logger.info("可以通过 'pip install onnxoptimizer' 安装优化工具")
    except Exception as e:
        logger.error(f"模型优化失败: {str(e)}")

def benchmark_models():
    """基准测试PyTorch vs ONNX性能"""
    
    try:
        import time
        
        onnx_dir = os.path.join(MODELS_DIR, 'onnx')
        facenet_onnx_path = os.path.join(onnx_dir, 'facenet_vggface2.onnx')
        
        if not os.path.exists(facenet_onnx_path):
            logger.error("ONNX模型不存在，请先运行转换")
            return
        
        # 加载模型
        pytorch_model = InceptionResnetV1(pretrained='vggface2')
        pytorch_model.eval()
        
        ort_session = onnxruntime.InferenceSession(facenet_onnx_path)
        
        # 准备测试数据
        test_data = np.random.randn(1, 3, 160, 160).astype(np.float32)
        test_tensor = torch.from_numpy(test_data)
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = pytorch_model(test_tensor)
            _ = ort_session.run(None, {ort_session.get_inputs()[0].name: test_data})
        
        # PyTorch基准测试
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = pytorch_model(test_tensor)
        pytorch_time = (time.time() - start_time) / num_runs
        
        # ONNX基准测试
        start_time = time.time()
        for _ in range(num_runs):
            _ = ort_session.run(None, {ort_session.get_inputs()[0].name: test_data})
        onnx_time = (time.time() - start_time) / num_runs
        
        speedup = pytorch_time / onnx_time
        
        logger.info("=== 性能基准测试结果 ===")
        logger.info(f"PyTorch 平均推理时间: {pytorch_time*1000:.2f} ms")
        logger.info(f"ONNX 平均推理时间: {onnx_time*1000:.2f} ms")
        logger.info(f"加速比: {speedup:.2f}x")
        
        if speedup > 1:
            logger.info("✅ ONNX推理速度更快！")
        else:
            logger.info("⚠️ PyTorch推理速度更快，可能需要进一步优化")
        
    except Exception as e:
        logger.error(f"性能基准测试失败: {str(e)}")

def main():
    """主函数"""
    
    logger.info("开始模型转换...")
    
    # 转换FaceNet
    facenet_path = convert_facenet_to_onnx()
    
    # 转换MTCNN
    mtcnn_paths = convert_mtcnn_to_onnx()
    
    # 优化模型
    optimize_onnx_models()
    
    # 性能基准测试
    benchmark_models()
    
    logger.info("模型转换完成！")
    
    # 输出转换结果信息
    onnx_dir = os.path.join(MODELS_DIR, 'onnx')
    logger.info(f"\n转换后的ONNX模型保存在: {onnx_dir}")
    logger.info("可用的ONNX模型:")
    for file in os.listdir(onnx_dir):
        if file.endswith('.onnx'):
            file_path = os.path.join(onnx_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"  - {file} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    main()
