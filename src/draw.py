"""画面和侧栏。中文用系统字体绘制。"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FONT_CANDIDATES
from .decide import Decision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

HAND_LABELS = {"peace": "比耶", "finger_heart": "比心"}
PANEL_WIDTH = 430
BG = (22, 24, 30)
CARD = (38, 42, 54)
CARD_ON = (48, 78, 112)
TEXT = (236, 238, 242)
MUTED = (168, 174, 186)
GREEN = (74, 214, 140)
RED = (255, 107, 107)
AMBER = (245, 184, 86)


def compose(
    frame: np.ndarray,
    view,
    decision: Decision,
    prompts: list[str],
    selected: int,
    summary: str,
    enrolled: list[str],
    fps: float,
) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], int]]]:
    canvas = _fit_height(frame, 720)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    pen = ImageDraw.Draw(image)
    _draw_scene(pen, image.size, frame.shape[1], frame.shape[0], view, decision.passed)
    panel = Image.new("RGB", (PANEL_WIDTH, image.height), BG)
    panel_pen = ImageDraw.Draw(panel)
    buttons = _draw_panel(panel_pen, panel.size, decision, prompts, selected, summary, enrolled, fps)
    combined = Image.new("RGB", (image.width + PANEL_WIDTH, image.height))
    combined.paste(image, (0, 0))
    combined.paste(panel, (image.width, 0))
    shifted = [((rect[0] + image.width, rect[1], rect[2] + image.width, rect[3]), index) for rect, index in buttons]
    return cv2.cvtColor(np.asarray(combined), cv2.COLOR_RGB2BGR), shifted


def overlay_lines(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    pen = ImageDraw.Draw(image)
    font = _font(28)
    block_h = 24 + 42 * max(1, len(lines))
    pen.rectangle((0, 0, image.width, block_h), fill=(22, 24, 30, 190))
    y = 16
    for line in lines:
        pen.text((24, y), line, font=font, fill=TEXT)
        y += 42
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _draw_scene(pen: ImageDraw.ImageDraw, size, src_w: int, src_h: int, view, passed: bool) -> None:
    width, height = size
    sx = width / src_w
    sy = height / src_h
    font = _font(26)
    for face in view.faces:
        x1, y1, x2, y2 = face.bbox
        box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        if face.subject and passed:
            color = GREEN
        elif face.name:
            color = AMBER
        else:
            color = RED
        pen.rectangle(box, outline=color, width=4 if face.subject else 2)
        label = face.name or "未注册"
        if face.subject:
            label = "核验 " + label
        label = f"{label}  {face.score:.2f}"
        pen.text((box[0], max(0, box[1] - 34)), label, font=font, fill=color)

    hand_font = _font(24)
    for hand in view.hands:
        color = GREEN if hand.label else (120, 180, 255)
        for start, end in HAND_CONNECTIONS:
            p1 = hand.points[start]
            p2 = hand.points[end]
            pen.line((p1[0] * width, p1[1] * height, p2[0] * width, p2[1] * height), fill=color, width=3)
        if hand.label:
            wrist = hand.points[0]
            pen.text((wrist[0] * width + 8, wrist[1] * height - 28), HAND_LABELS[hand.label], font=hand_font, fill=color)


def _draw_panel(pen, size, decision, prompts, selected, summary, enrolled, fps) -> list[tuple[tuple[int, int, int, int], int]]:
    width, height = size
    title = _font(32)
    body = _font(22)
    small = _font(18)
    enrolled_text = "、".join(enrolled[:4]) if enrolled else "无"
    pen.text((24, 16), "文本条件核验", font=title, fill=TEXT)
    pen.text((24, 56), f"{fps:.0f} 帧/秒  ·  已注册 {enrolled_text}", font=small, fill=MUTED)

    y = 92
    buttons = []
    button_font = _font(20)
    for index, prompt in enumerate(prompts):
        if y + 34 > height - 220:
            break
        rect = (24, y, width - 24, y + 34)
        fill = CARD_ON if index == selected else CARD
        pen.rounded_rectangle(rect, radius=8, fill=fill)
        prefix = f"{index + 1}  " if index < 9 else ""
        pen.text((34, y + 5), prefix + prompt, font=button_font, fill=TEXT)
        buttons.append((rect, index))
        y += 40

    y += 10
    summary_color = RED if decision.message not in {"通过", "不通过"} else TEXT
    pen.text((24, y), summary, font=body, fill=summary_color)
    y += 36
    if decision.checks:
        for item in decision.checks:
            color = GREEN if item.ok else RED
            mark = "通过" if item.ok else "未过"
            pen.text((24, y), f"{item.title}  {mark}  {item.detail}", font=small, fill=color)
            y += 28
    else:
        pen.text((24, y), "这句要求无法使用", font=small, fill=RED)
        y += 28

    banner = "通过" if decision.passed else "不通过"
    banner_color = GREEN if decision.passed else RED
    pen.rounded_rectangle((24, y + 6, width - 24, y + 58), radius=12, fill=banner_color)
    pen.text((40, y + 12), banner, font=title, fill=(20, 24, 28))
    y += 72
    pen.text((24, y), f"表情  {decision.expression_text}", font=small, fill=MUTED)
    y += 26
    pen.text((24, y), f"手势  {decision.gesture_text}", font=small, fill=MUTED)
    y += 30
    pen.text((24, y), "按 1-6 或点击句子，Q 退出", font=small, fill=MUTED)
    return buttons


def _fit_height(frame: np.ndarray, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    width = max(1, int(src_w * height / src_h))
    return cv2.resize(frame, (width, height))


def _font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()

