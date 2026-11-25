"""注册、查看或删除人脸。"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import FACE_DB_DIR
from src.draw import overlay_lines
from src.gallery import Gallery, safe_person_name
from src.identity import IdentityEngine

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    args = _parse_args()
    gallery = Gallery.load()
    if args.list:
        counts = gallery.counts()
        if not counts:
            print("人脸库是空的。")
            return
        for name, count in counts.items():
            print(f"{name}  {count} 张")
        return
    if args.remove:
        name = safe_person_name(args.remove)
        removed = gallery.remove(name)
        gallery.save()
        print(f"已删除 {name} 的 {removed} 条特征。")
        return
    if not args.name:
        raise SystemExit("请用 --name 指定要注册的人，或使用 --list / --remove。")
    name = safe_person_name(args.name)
    if args.images:
        embeddings = _from_images(args.images)
    else:
        embeddings = _from_camera(name, args.camera, args.count)
    gallery.replace(name, embeddings)
    gallery.save()
    print(f"已注册 {name}，共 {len(embeddings)} 张。重新打开演示即可使用。")


def _from_images(folder: Path) -> list[np.ndarray]:
    if not folder.is_dir():
        raise SystemExit(f"找不到图片目录：{folder}")
    paths = [path for path in sorted(folder.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        raise SystemExit(f"{folder} 里没有 jpg 或 png。")
    print("正在加载模型，第一次运行会从网络下载。")
    engine = IdentityEngine()
    vectors = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"无法读取，已跳过：{path.name}")
            continue
        faces = engine.detect(image)
        if not faces:
            print(f"没有检测到人脸，已跳过：{path.name}")
            continue
        vectors.append(faces[0].embedding)
        print(f"已使用 {path.name}")
    if not vectors:
        raise SystemExit("这些图片里没有可用的人脸。")
    return vectors


def _from_camera(name: str, camera_id: int, count: int) -> list[np.ndarray]:
    print("正在加载模型，第一次运行会从网络下载。")
    engine = IdentityEngine()
    camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        raise SystemExit(f"打不开摄像头 {camera_id}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    saved: list[np.ndarray] = []
    crops: list[np.ndarray] = []
    minimum = min(3, count)
    cooldown = 0
    window = "enroll"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit("读不到摄像头画面。")
            frame = cv2.flip(frame, 1)
            faces = engine.detect(frame)
            for face in faces[1:]:
                cv2.rectangle(frame, face.bbox[:2], face.bbox[2:], (80, 80, 80), 1)
            if faces:
                face = faces[0]
                cv2.rectangle(frame, face.bbox[:2], face.bbox[2:], (74, 214, 140), 2)
            lines = [
                f"正在注册：{name}    {len(saved)}/{count}",
                "空格保存当前人脸，回车完成，Q 放弃",
                "请保持正脸。画面里有多个人时，注册最大的那张脸。",
            ]
            if len(saved) < minimum:
                lines.append(f"至少保存 {minimum} 张才能完成。")
            cv2.imshow(window, overlay_lines(frame, lines))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise SystemExit("已放弃注册，人脸库没有改动。")
            if cooldown:
                cooldown -= 1
            if key == ord(" ") and faces and cooldown == 0:
                face = faces[0]
                x1, y1, x2, y2 = face.bbox
                crops.append(frame[y1:y2, x1:x2].copy())
                saved.append(face.embedding)
                cooldown = 8
                print(f"已保存第 {len(saved)} 张")
                if len(saved) >= count:
                    break
            if key in (13, 10) and len(saved) >= minimum:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    _store_crops(name, crops)
    return saved


def _store_crops(name: str, crops: list[np.ndarray]) -> None:
    folder = FACE_DB_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    for index, crop in enumerate(crops, start=1):
        if crop.size == 0:
            continue
        path = folder / f"{index:02d}.jpg"
        ok, encoded = cv2.imencode(".jpg", crop)
        if ok:
            encoded.tofile(str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="注册人脸")
    parser.add_argument("--name", help="要注册的名字")
    parser.add_argument("--images", type=Path, help="从该文件夹的照片注册，而不是打开摄像头")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--count", type=int, default=5, help="摄像头注册时保存的张数")
    parser.add_argument("--list", action="store_true", help="列出现有的人")
    parser.add_argument("--remove", help="删除这个人的特征")
    return parser.parse_args()


if __name__ == "__main__":
    main()
