"""把身份、表情、手势合成通过或不通过。"""

from dataclasses import dataclass, field

from .config import EXPRESSION_LABELS, EXPRESSION_THRESHOLD, GESTURE_LABELS, IDENTITY_THRESHOLD
from .expression import dominant_expression, empty_scores
from .gesture import gesture_satisfied
from .prompt import Requirement


@dataclass
class Observation:
    has_face: bool
    name: str | None
    score: float
    expression: dict[str, float] = field(default_factory=empty_scores)
    peace_count: int = 0
    heart_count: int = 0


@dataclass
class Check:
    title: str
    ok: bool
    detail: str


@dataclass
class Decision:
    passed: bool
    checks: list[Check]
    expression_text: str
    gesture_text: str
    message: str


def decide(
    observation: Observation,
    requirement: Requirement,
    identity_threshold: float = IDENTITY_THRESHOLD,
    gallery_empty: bool = False,
) -> Decision:
    expression_text = _expression_text(observation.expression)
    gesture_text = f"比耶 {observation.peace_count} 只，比心 {observation.heart_count} 只"
    if requirement.error:
        return Decision(False, [], expression_text, gesture_text, requirement.error)

    checks = [_identity_check(observation, identity_threshold, gallery_empty)]
    if requirement.expression:
        label = EXPRESSION_LABELS[requirement.expression]
        score = observation.expression.get(requirement.expression, 0.0)
        threshold = EXPRESSION_THRESHOLD[requirement.expression]
        ok = observation.has_face and score >= threshold
        checks.append(Check("表情", ok, f"需要{label}  {score:.2f}"))
    if requirement.gesture:
        label = GESTURE_LABELS[requirement.gesture]
        ok = gesture_satisfied(requirement.gesture, observation.peace_count, observation.heart_count)
        checks.append(Check("手势", ok, f"需要{label}"))

    passed = all(item.ok for item in checks)
    message = "通过" if passed else "不通过"
    return Decision(passed, checks, expression_text, gesture_text, message)


def _identity_check(observation: Observation, threshold: float, gallery_empty: bool) -> Check:
    if gallery_empty:
        return Check("身份", False, "请先注册")
    if not observation.has_face:
        return Check("身份", False, "未检测到人脸")
    if observation.name:
        return Check("身份", True, f"{observation.name}  {observation.score:.2f}")
    return Check("身份", False, f"相似度 {observation.score:.2f}，未达 {threshold:.2f}")


def _expression_text(scores: dict[str, float]) -> str:
    name, score = dominant_expression(scores)
    labels = {"neutral": "中性", **EXPRESSION_LABELS}
    return f"{labels[name]} {score:.2f}"
