from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

from core.schemas import UserProfile, WardrobeItem
from core.scoring_engine import weighted_score


@dataclass(frozen=True)
class OutfitCandidate:
    items: tuple[WardrobeItem, ...]
    score: float
    components: dict[str, float]
    reasons: tuple[str, ...]


def recommend_outfits(
    profile: UserProfile,
    wardrobe_items: Iterable[WardrobeItem] | None = None,
    occasion: str | None = None,
    top_k: int = 5,
) -> list[OutfitCandidate]:
    """Generate explainable outfit candidates using only supplied wardrobe items."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    items = [item for item in (wardrobe_items if wardrobe_items is not None else profile.wardrobe_items) if item and item.item_id]
    target_occasion = occasion or profile.occasion or "Everyday"
    candidates: list[OutfitCandidate] = []
    for outfit in _candidate_combinations(items):
        candidate = score_outfit(profile, outfit, target_occasion)
        candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:top_k]


def score_outfit(
    profile: UserProfile,
    outfit: Sequence[WardrobeItem],
    occasion: str | None = None,
) -> OutfitCandidate:
    """Score one outfit with normalized, inspectable components in [0, 1]."""
    target_occasion = occasion or profile.occasion or "Everyday"
    items = tuple(item for item in outfit if item and item.item_id)
    categories = {_category(item) for item in items}
    category_score = _category_compatibility(categories)
    color_score = _color_compatibility(items, profile)
    occasion_score = _occasion_compatibility(items, target_occasion)
    style_score = _style_compatibility(profile, items)
    body_shape_score = _body_shape_compatibility(profile.body_shape, items)
    comfort_score = _comfort_compatibility(items, target_occasion)
    components = {
        "color_compatibility": color_score,
        "body_shape_compatibility": body_shape_score,
        "occasion_compatibility": occasion_score,
        "comfort_practicality": comfort_score,
        "style_relevance": style_score,
        "style_compatibility": style_score,
        "category_compatibility": category_score,
    }
    score = weighted_score(components)
    reasons = _explain(components, categories, target_occasion, items)
    return OutfitCandidate(items=items, score=score, components=components, reasons=reasons)


def _candidate_combinations(items: Sequence[WardrobeItem]):
    tops = [item for item in items if _category(item) in {"top", "outerwear"}]
    bottoms = [item for item in items if _category(item) in {"bottom"}]
    dresses = [item for item in items if _category(item) == "dress"]
    shoes = [item for item in items if _category(item) == "shoes"]
    accessories = [item for item in items if _category(item) == "accessory"]
    layers = [item for item in items if _category(item) == "outerwear"]

    yielded = False
    for dress in dresses:
        shoe_options = shoes or [None]
        for shoe in shoe_options:
            for accessory in accessories[:1] or [None]:
                yielded = True
                yield tuple(item for item in (dress, shoe, accessory) if item)
    for top in tops:
        for bottom in bottoms:
            shoe_options = shoes or [None]
            for shoe in shoe_options:
                for layer in layers[:1] or [None]:
                    yielded = True
                    yield tuple(item for item in (top, bottom, shoe, layer) if item)
    if not yielded:
        for item in items:
            yield (item,)


def _category(item: WardrobeItem) -> str:
    value = (item.category or "").strip().lower()
    aliases = {
        "top": "top", "shirt": "top", "t-shirt": "top", "blouse": "top",
        "bottom": "bottom", "trousers": "bottom", "pants": "bottom", "jeans": "bottom", "shorts": "bottom", "skirt": "bottom",
        "dress": "dress", "shoes": "shoes", "shoe": "shoes",
        "outerwear": "outerwear", "jacket": "outerwear", "coat": "outerwear",
        "accessory": "accessory", "bag": "accessory", "handbag": "accessory",
    }
    return aliases.get(value, value or "uncategorized")


def _category_compatibility(categories: set[str]) -> float:
    if "dress" in categories:
        return 1.0 if categories <= {"dress", "shoes", "accessory", "outerwear"} else 0.55
    if {"top", "bottom"}.issubset(categories):
        return 1.0 if "shoes" in categories else 0.82
    if len(categories) == 1 and categories <= {"shoes", "accessory"}:
        return 0.25
    return 0.45


def _color_compatibility(items: Sequence[WardrobeItem], profile: UserProfile | None = None) -> float:
    colors = [item.color.strip().lower() for item in items if item.color]
    if len(colors) < 2:
        base = 0.55
        preferred = {value.lower() for value in (profile.preferred_colors if profile else [])}
        return min(1.0, base + 0.12) if any(color in preferred for color in colors) else base
    neutral = {"black", "white", "cream", "ivory", "grey", "gray", "navy", "beige", "brown", "denim"}
    if all(color in neutral for color in colors):
        base = 0.95
    elif any(color in neutral for color in colors):
        base = 0.85
    else:
        base = 0.7 if len(set(colors)) <= 2 else 0.45
    preferred = {value.lower() for value in (profile.preferred_colors if profile else [])}
    return min(1.0, base + 0.08) if any(color in preferred for color in colors) else base


def _occasion_compatibility(items: Sequence[WardrobeItem], occasion: str) -> float:
    target = occasion.lower()
    categories = {_category(item) for item in items}
    if target in {"wedding", "traditional", "party", "dinner"} and ("dress" in categories or "outerwear" in categories):
        return 0.9
    if target in {"formal", "presentation", "interview"} and categories & {"top", "bottom", "dress", "outerwear"}:
        return 0.92
    if target in {"college", "casual", "travel", "outdoor"} and categories & {"top", "bottom", "dress"}:
        return 0.86
    if target == "work" and categories & {"top", "bottom", "dress", "outerwear"}:
        return 0.85
    if target in {"travel", "weekend", "everyday"}:
        return 0.8 if categories & {"top", "bottom", "dress"} else 0.5
    return 0.6


def _style_compatibility(profile: UserProfile, items: Sequence[WardrobeItem]) -> float:
    preferences = {value.lower() for value in profile.style_preferences}
    if not preferences:
        return 0.6
    text = " ".join(f"{item.name} {item.pattern or ''} {item.category}".lower() for item in items)
    matches = sum(preference in text for preference in preferences)
    return min(1.0, 0.55 + matches * 0.15)


def _body_shape_compatibility(body_shape: str | None, items: Sequence[WardrobeItem]) -> float:
    shape = (body_shape or "").lower()
    text = " ".join(f"{item.name} {item.subcategory or ''} {item.fit or ''} {item.style or ''}".lower() for item in items)
    rules = {
        "pear": ("a-line", "fit & flare", "wide-leg", "straight", "balanced"),
        "hourglass": ("wrap", "waist", "fitted", "semi-fitted", "high-waist"),
        "rectangle": ("layer", "waist", "a-line", "fit & flare"),
        "apple": ("a-line", "straight", "relaxed", "structured"),
        "inverted triangle": ("a-line", "wide-leg", "balanced"),
    }
    matches = sum(term in text for term in rules.get(shape, ()))
    return min(1.0, 0.72 + matches * 0.08) if shape else 0.72


def _comfort_compatibility(items: Sequence[WardrobeItem], occasion: str) -> float:
    text = " ".join(f"{item.name} {item.fit or ''} {item.style or ''}".lower() for item in items)
    if occasion.lower() in {"college", "travel", "outdoor"}:
        return 0.95 if any(term in text for term in ("comfortable", "relaxed", "flat", "sneaker")) else 0.78
    return 0.9 if any(term in text for term in ("semi-fitted", "structured", "comfortable")) else 0.78


def _explain(components: dict[str, float], categories: set[str], occasion: str, items: Sequence[WardrobeItem]) -> tuple[str, ...]:
    reasons = []
    if components["category_compatibility"] >= 0.8:
        reasons.append("The categories form a complete outfit structure.")
    else:
        reasons.append("The wardrobe has limited category coverage for a full outfit.")
    if components["color_compatibility"] >= 0.8:
        reasons.append("The colors use compatible neutral or accent relationships.")
    if components["occasion_compatibility"] >= 0.8:
        reasons.append(f"The combination suits the {occasion.lower()} occasion.")
    if components["style_compatibility"] >= 0.7:
        reasons.append("The available style preferences align with the selected pieces.")
    if components.get("body_shape_compatibility", 0) >= 0.8:
        reasons.append("The silhouette tags align with the detected body-shape styling guidance.")
    return tuple(reasons)
