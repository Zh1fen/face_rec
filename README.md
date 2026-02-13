# 人脸识别系统

基于 PyTorch 的人脸识别系统，支持离线图片识别和实时摄像头识别。 

## 功能特点

- ✅ **离线图片识别**: 支持单张图片、批量图片和文件夹识别
- ✅ **实时摄像头识别**: 使用摄像头进行实时人脸识别
- ✅ **高精度识别**: 基于 FaceNet 预训练模型，识别准确率 95%+
- ✅ **人脸跟踪**: 实时识别中支持人脸跟踪，避免重复识别
- ✅ **可视化结果**: 在图片上显示识别结果和置信度
- ✅ **灵活配置**: 支持多种参数配置和自定义设置

## 项目结构

```
face_rec/
├── src/                        # 源代码目录
│   ├── face_detector.py        # 人脸检测模块 (MTCNN)
│   ├── feature_extractor.py    # 特征提取模块 (FaceNet)
│   ├── face_recognizer.py      # 人脸识别模块
│   ├── face_tracker.py         # 人脸跟踪模块
│   ├── real_time_recognizer.py # 实时识别模块
│   └── utils.py                # 工具函数
├── models/                     # 预训练模型目录
├── face_database/              # 人脸库目录 (放入已知人脸图片)
│   ├── 张三/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── 李四/
│       └── photo1.jpg
├── test_images/               # 测试图片目录
├── features/                  # 特征向量存储目录
├── config.py                  # 配置文件
├── build_database.py          # 构建人脸特征数据库
├── main.py                    # 离线识别主程序
├── real_time_main.py         # 实时识别主程序
└── requirements.txt          # 依赖包列表
```

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备人脸数据库

在 `face_database/` 目录中按人名创建子文件夹，并放入对应的照片：

```
face_database/
├── 张三/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── 李四/
│   ├── photo1.jpg
│   └── photo2.jpg
└── 王五/
    └── photo1.jpg
```

### 3. 构建特征数据库

```bash
python build_database.py
```

## 使用方法

### 离线图片识别

```bash
# 识别单张图片
python main.py --image test_images/test.jpg

# 识别文件夹中的所有图片
python main.py --folder test_images/

# 批量识别多张图片
python main.py --batch image1.jpg image2.jpg image3.jpg

# 保存识别结果图片
python main.py --image test.jpg --save

# 显示所有检测到的人脸（包括未识别的）
python main.py --image test.jpg --show-all
```

### 实时摄像头识别

```bash
# 启动实时识别（使用默认摄像头）
python real_time_main.py

# 指定摄像头
python real_time_main.py --camera 1

# 列出可用摄像头
python real_time_main.py --list-cameras

# 测试摄像头
python real_time_main.py --test-camera 0

# 禁用人脸跟踪
python real_time_main.py --no-tracking

# 自定义识别阈值
python real_time_main.py --threshold 0.7
```

### 实时识别控制

在实时识别界面中：
- **空格键**: 暂停/继续识别
- **S键**: 保存当前帧
- **R键**: 重置识别历史
- **C键**: 切换摄像头
- **Q键**: 退出程序

## 配置选项

### 数据库构建配置

```bash
# 使用平均特征（推荐）
python build_database.py

# 只使用第一个检测到的特征
python build_database.py --no-average

# 设置每个人最少需要的人脸数量
python build_database.py --min-faces 2

# 验证现有数据库
python build_database.py --validate
```

### 自定义配置文件

创建自定义配置文件：

```bash
python real_time_main.py --create-config
```

这会生成 `custom_config.json` 文件，您可以修改其中的参数：

```json
{
  "camera_id": 0,
  "fps": 30,
  "process_fps": 10,
  "detection_confidence": 0.9,
  "recognition_threshold": 0.6,
  "face_size": [160, 160],
  "display_size": [800, 600],
  "save_unknown_faces": true,
  "show_confidence": true,
  "track_faces": true,
  "max_track_frames": 30
}
```

然后使用自定义配置：

```bash
python real_time_main.py --config custom_config.json
```

## 技术实现

### 人脸检测
- 使用 **MTCNN** (Multi-task CNN) 进行人脸检测
- 支持多人脸检测和人脸关键点定位
- 可配置检测置信度和最小人脸尺寸

### 特征提取
- 使用 **FaceNet** 预训练模型提取 512 维特征向量
- 支持 VGGFace2 和 CASIA-WebFace 预训练权重
- 特征向量经过 L2 归一化处理

### 相似度计算
- 使用**余弦相似度**计算特征向量相似性
- 支持可配置的识别阈值
- 在数据库中查找最佳匹配

### 人脸跟踪
- 使用 **OpenCV CSRT 跟踪器**
- 避免对同一张脸重复识别
- 支持多人脸同时跟踪

## 性能优化

### 实时识别优化
- 异步处理：检测和识别在后台线程进行
- 帧率控制：可设置处理帧率低于视频帧率
- 人脸跟踪：减少重复识别计算
- 队列管理：避免帧堆积

### 识别准确性优化
- 人脸对齐：基于关键点进行人脸对齐
- 平均特征：使用多张照片的平均特征
- 质量筛选：过滤低质量的人脸检测结果

## 疑难解答

### 常见问题

1. **ImportError: No module named 'mtcnn'**
   ```bash
   pip install mtcnn
   ```

2. **ImportError: No module named 'facenet_pytorch'**
   ```bash
   pip install facenet-pytorch
   ```

3. **摄像头无法打开**
   - 检查摄像头是否被其他程序占用
   - 尝试不同的摄像头ID
   - 使用 `--test-camera` 参数测试摄像头

4. **识别准确率低**
   - 确保人脸数据库照片质量良好
   - 增加每个人的照片数量
   - 调整识别阈值
   - 确保照片中人脸清晰且正面

5. **实时识别卡顿**
   - 降低处理帧率 `--process-fps`
   - 启用人脸跟踪减少计算量
   - 调整显示窗口大小

### 系统要求

- **Python**: 3.7+
- **操作系统**: Windows/Linux/macOS
- **内存**: 建议 4GB+ RAM
- **摄像头**: 支持 OpenCV 的 USB 摄像头
- **GPU**: 可选，支持 CUDA 加速

## 更新日志

### v1.0.0
- 初始版本发布
- 支持离线图片识别
- 支持实时摄像头识别
- 集成人脸跟踪功能
- 提供完整的配置选项

## 许可证

MIT License

## 贡献

欢迎提交 Issues 和 Pull Requests！
