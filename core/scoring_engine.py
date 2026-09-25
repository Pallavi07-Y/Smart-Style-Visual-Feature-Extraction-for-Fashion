from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreWeights:
    color: int = 25
    body_shape: int = 25
    occasion: int = 25
    comfort: int = 10
    style: int = 15

    @property
    def total(self) -> int:
        return self.color + self.body_shape + self.occasion + self.comfort + self.style


DEFAULT_WEIGHTS = ScoreWeights()


def weighted_score(components: dict[str, float], weights: ScoreWeights = DEFAULT_WEIGHTS) -> float:
    """Combine normalized deterministic components using the product's published weights."""
    values = {
        "color": components.get("color_compatibility", 0.0),
        "body_shape": components.get("body_shape_compatibility", 0.0),
        "occasion": components.get("occasion_compatibility", 0.0),
        "comfort": components.get("comfort_practicality", 0.0),
        "style": components.get("style_relevance", components.get("style_compatibility", 0.0)),
    }
    points = (
        values["color"] * weights.color
        + values["body_shape"] * weights.body_shape
        + values["occasion"] * weights.occasion
        + values["comfort"] * weights.comfort
        + values["style"] * weights.style
    )
    return round(points / weights.total, 4) if weights.total else 0.0