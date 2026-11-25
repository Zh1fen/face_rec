"""由面部 blendshape 系数得到微笑、惊讶、生气的分数。"""

EXPRESSION_KEYS = ("smile", "surprise", "anger")


def scores_from_map(values: dict[str, float]) -> dict[str, float]:
    smile = _avg(values, "mouthSmileLeft", "mouthSmileRight")
    eye_wide = _avg(values, "eyeWideLeft", "eyeWideRight")
    brow_down = _avg(values, "browDownLeft", "browDownRight")
    frown = _avg(values, "mouthFrownLeft", "mouthFrownRight")
    surprise = (
        0.45 * values.get("jawOpen", 0.0)
        + 0.30 * eye_wide
        + 0.25 * values.get("browInnerUp", 0.0)
    )
    anger = 0.65 * brow_down + 0.35 * frown
    return {"smile": float(smile), "surprise": float(surprise), "anger": float(anger)}


def scores_from_categories(categories) -> dict[str, float]:
    if not categories:
        return empty_scores()
    values = {item.category_name: float(item.score) for item in categories}
    return scores_from_map(values)


def empty_scores() -> dict[str, float]:
    return {key: 0.0 for key in EXPRESSION_KEYS}


def dominant_expression(scores: dict[str, float], minimum: float = 0.25) -> tuple[str, float]:
    filled = empty_scores()
    filled.update(scores)
    name = max(EXPRESSION_KEYS, key=filled.get)
    score = filled[name]
    if score < minimum:
        return "neutral", score
    return name, score


def _avg(values: dict[str, float], *names: str) -> float:
    return sum(values.get(name, 0.0) for name in names) / len(names)
