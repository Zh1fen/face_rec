# 文本条件人脸核验

摄像头里要同时满足文字提出的条件才算通过：是注册过的那个人，并且表情、手势和这句话一致。一句话可以只写表情、只写动作，或者两样一起写。

例如「请微笑并用单手比耶」会检查三件事：

- 身份是已注册的人
- 表情是微笑
- 刚好有一只手在比耶

本人只微笑、没有比耶，结果是不通过。换一个人即使做对了动作，也是不通过。

## 准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果系统没有 `python3-venv`，可以用 [uv](https://docs.astral.sh/uv/)：

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

第一次运行会下载四个人脸和手部模型，需要能访问 GitHub 和 Google 的模型地址。

## 注册

摄像头前按空格保存正脸，默认 5 张，至少 3 张后可以按回车结束：

```bash
python enroll.py --name 张三
```

也可以直接用一个文件夹里的照片。每个人单独放一个文件夹：

```bash
python enroll.py --name 张三 --images face_database/张三
```

```bash
python enroll.py --list
python enroll.py --remove 张三
```

## 演示

```bash
python app.py
```

右侧列出这些句子，点击或按数字键 `1` 到 `6` 切换：

- 请微笑
- 请做出惊讶的表情
- 请单手比耶
- 请双手比耶
- 请微笑并用单手比耶
- 请比心

按 `Q` 退出。句子写在 `prompts.json` 里，改完重新运行即可。

指定启动时的第一句：

```bash
python app.py --prompt "请微笑并用单手比耶"
```

核验的是画面里最大的那张脸。手要完整出现在画面中。比心指的是单手用拇指和食指捏出的小心心。

能识别的说法：

- 表情：微笑、开心、惊讶、吃惊、生气、皱眉
- 手势：单手比耶、双手比耶、剪刀手、比心、比爱心

「不要笑，请比耶」这种否定会生效。一句话里放两种表情或两种手势会提示改句子，不会各算各的。

## 测试

不打开摄像头，只检查文字解析、手势规则和通过条件：

```bash
python -m unittest discover -s tests
```

## 目录

- `app.py`：摄像头演示
- `enroll.py`：注册人脸
- `prompts.json`：演示用的句子
- `src/prompt.py`：把句子拆成表情和手势
- `src/identity.py`：YuNet 找脸，SFace 认人
- `src/vision.py`：MediaPipe 看表情和手
- `src/gesture.py`：比耶、比心
- `data/`：注册后的特征，不进 git

人脸特征用 `npy` 和 `json` 保存。不要把旧的 pickle 特征文件拷回来用。
