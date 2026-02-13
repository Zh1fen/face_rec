@echo off
echo 人脸识别系统快速启动
echo ================================

echo.
echo 1. 构建人脸数据库
echo python build_database.py
echo.

echo 2. 离线图片识别
echo python main.py --image test_images/your_image.jpg
echo.

echo 3. 实时摄像头识别  
echo python real_time_main.py
echo.

echo 4. 测试摄像头
echo python real_time_main.py --list-cameras
echo.

pause
