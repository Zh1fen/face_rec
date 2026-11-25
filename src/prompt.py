"""把中文要求拆成表情和手势。两项可以同时出现。"""

from dataclasses import dataclass

from .config import EXPRESSION_LABELS, GESTURE_LABELS

# 长词写在前面。匹配时还会再按字数排序，避免「比耶」吃掉「双手比耶」。
GESTURE_PATTERNS = [
    ("peace_two", ["双手比耶", "两只手比耶", "双手剪刀手", "双手耶"]),
    ("peace_one", ["单手比耶", "一只手比耶", "单手剪刀手", "比耶", "剪刀手"]),
    ("finger_heart", ["单手比心", "比爱心", "比心", "爱心"]),
]

EXPRESSION_PATTERNS = [
    ("surprise", ["惊讶", "吃惊", "惊喜"]),
    ("anger", ["生气", "愤怒", "发怒", "皱眉"]),
    ("smile", ["微笑", "开心", "高兴", "笑一个", "笑"]),
]


@dataclass
class Requirement:
    expression: str | None
    gesture: str | None
    error: str | None

    def summary(self) -> str:
        if self.error:
            return self.error
        parts = []
        if self.expression:
            parts.append(EXPRESSION_LABELS[self.expression])
        if self.gesture:
            parts.append(GESTURE_LABELS[self.gesture])
        return " + ".join(parts)


def parse(text: str) -> Requirement:
    raw = " ".join(text.strip().split())
    expressions = _find(raw, EXPRESSION_PATTERNS)
    gestures = _find(raw, GESTURE_PATTERNS)
    errors = []
    if len(expressions) > 1:
        errors.append("这句话里有两种表情，请只留一种")
    if len(gestures) > 1:
        errors.append("这句话里有两种手势，请只留一种")
    if not expressions and not gestures:
        errors.append("没有识别到表情或手势，可以试试「请微笑并用单手比耶」")
    if errors:
        return Requirement(None, None, "；".join(errors))
    return Requirement(
        expressions[0] if expressions else None,
        gestures[0] if gestures else None,
        None,
    )


def _find(text: str, patterns: list[tuple[str, list[str]]]) -> list[str]:
    items = []
    for label, phrases in patterns:
        for phrase in phrases:
            items.append((len(phrase), label, phrase))
    items.sort(key=lambda item: item[0], reverse=True)

    occupied = [False] * len(text)
    found: list[str] = []
    for _, label, phrase in items:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index < 0:
                break
            end = index + len(phrase)
            if any(occupied[index:end]):
                start = index + 1
                continue
            for cursor in range(index, end):
                occupied[cursor] = True
            if not _negated(text, index) and label not in found:
                found.append(label)
            start = end
    return found


def _negated(text: str, start: int) -> bool:
    if start >= 1 and text[start - 1] in "不别勿":
        return True
    if start >= 2 and text[start - 2:start] in {"不要", "别再"}:
        return True
    return False
