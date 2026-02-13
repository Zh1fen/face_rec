# 测试图片目录

请将要测试的图片放在此目录中。

## 使用方法

将测试图片放入此目录后，可以使用以下命令进行测试：

```bash
# 测试整个文件夹
python main.py --folder test_images/

# 或者使用快捷方式
python main.py --test
```

## 支持的图片格式

- .jpg / .jpeg
- .png
- .bmp
- .tiff

## 示例

假设您有以下测试图片：
- group_photo.jpg (集体照)
- single_person.jpg (单人照)
- family.png (家庭照)

运行测试命令后，系统会识别每张图片中的人脸，并显示识别结果。
