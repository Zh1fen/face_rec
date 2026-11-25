"""用最近几帧的多数结果稳住画面上的判断。"""

from collections import Counter, deque

import numpy as np

from .decide import Observation
from .expression import empty_scores


class FrameSmoother:
    def __init__(self, window: int = 5):
        self.faces: deque[bool] = deque(maxlen=window)
        self.names: deque[str | None] = deque(maxlen=window)
        self.scores: deque[float] = deque(maxlen=window)
        self.expressions: deque[dict[str, float]] = deque(maxlen=window)
        self.peace: deque[int] = deque(maxlen=window)
        self.heart: deque[int] = deque(maxlen=window)

    def update(self, observation: Observation) -> Observation:
        self.faces.append(observation.has_face)
        self.names.append(observation.name if observation.has_face else None)
        self.scores.append(observation.score if observation.has_face else 0.0)
        self.expressions.append(dict(observation.expression) if observation.has_face else empty_scores())
        self.peace.append(observation.peace_count)
        self.heart.append(observation.heart_count)
        return self.snapshot()

    def snapshot(self) -> Observation:
        has_face = sum(self.faces) >= _majority(len(self.faces))
        accepted = [name for name in self.names if name]
        name = None
        if accepted:
            winner, votes = Counter(accepted).most_common(1)[0]
            if votes >= _majority(len(self.names)):
                name = winner
        if name:
            score = float(np.mean([value for person, value in zip(self.names, self.scores) if person == name]))
        elif self.scores:
            score = float(np.mean(self.scores))
        else:
            score = 0.0

        face_expressions = [item for item, present in zip(self.expressions, self.faces) if present]
        expression = empty_scores()
        if has_face and face_expressions:
            for key in expression:
                expression[key] = float(np.mean([item.get(key, 0.0) for item in face_expressions]))

        return Observation(
            has_face=has_face,
            name=name,
            score=score,
            expression=expression,
            peace_count=_mode(self.peace),
            heart_count=_mode(self.heart),
        )


def _majority(count: int) -> int:
    return max(1, (count + 1) // 2)


def _mode(values: deque[int]) -> int:
    if not values:
        return 0
    return Counter(values).most_common(1)[0][0]
