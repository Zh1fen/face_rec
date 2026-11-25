"""摄像头演示。文字里可以同时要求表情和动作。"""

import argparse
import json
import time

import cv2

from src.config import IDENTITY_THRESHOLD, PROMPTS_PATH, SMOOTH_WINDOW
from src.decide import decide
from src.draw import compose
from src.gallery import Gallery
from src.pipeline import Engine
from src.prompt import parse
from src.smooth import FrameSmoother


def main() -> None:
    args = _parse_args()
    prompts = _load_prompts()
    if args.prompt:
        prompts = [args.prompt] + [item for item in prompts if item != args.prompt]
    selected = 0
    gallery = Gallery.load()
    print("正在加载模型，第一次运行会从网络下载。")
    engine = Engine(gallery, args.threshold)
    smoother = FrameSmoother(SMOOTH_WINDOW)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        engine.close()
        raise SystemExit(f"打不开摄像头 {args.camera}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "demo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    click = {"point": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["point"] = (x, y)

    cv2.setMouseCallback(window, on_mouse)
    fps = 0.0
    last = time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("读不到摄像头画面，演示结束。")
                break
            frame = cv2.flip(frame, 1)
            raw = engine.analyze(frame)
            stable = smoother.update(raw.observation)
            for face in raw.faces:
                if face.subject:
                    face.name = stable.name
                    face.score = stable.score
            requirement = parse(prompts[selected])
            decision = decide(stable, requirement, args.threshold, gallery.empty())
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            canvas, buttons = compose(
                frame,
                raw,
                decision,
                prompts,
                selected,
                requirement.summary(),
                list(gallery.counts()),
                fps,
            )
            cv2.imshow(window, canvas)
            if click["point"] is not None:
                x, y = click["point"]
                click["point"] = None
                for rect, index in buttons:
                    if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                        selected = index
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("9"):
                index = key - ord("1")
                if index < len(prompts):
                    selected = index
    finally:
        camera.release()
        engine.close()
        cv2.destroyAllWindows()


def _load_prompts() -> list[str]:
    if not PROMPTS_PATH.exists():
        raise SystemExit("缺少 prompts.json")
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    prompts = [str(item).strip() for item in data if str(item).strip()]
    if not prompts:
        raise SystemExit("prompts.json 里没有可用的句子")
    return prompts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文本条件人脸核验演示")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--prompt", type=str, help="启动时优先使用的一句要求")
    parser.add_argument("--threshold", type=float, default=IDENTITY_THRESHOLD, help="身份余弦相似度阈值")
    return parser.parse_args()


if __name__ == "__main__":
    main()
