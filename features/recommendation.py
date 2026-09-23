from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

from core.schemas import UserProfile, WardrobeItem


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
    color_score = _color_compatibility(items)
    occasion_score = _occasion_compatibility(items, target_occasion)
    style_score = _style_compatibility(profile, items)
    availability_score = 1.0 if items and len({item.item_id for item in items}) == len(items) else 0.0
    components = {
        "category_compatibility": category_score,
        "color_compatibility": color_score,
        "occasion_compatibility": occasion_score,
        "style_compatibility": style_score,
        "wardrobe_availability": availability_score,
    }
    score = round(sum(components.values()) / len(components), 4)
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


def _color_compatibility(items: Sequence[WardrobeItem]) -> float:
    colors = [item.color.strip().lower() for item in items if item.color]
    if len(colors) < 2:
        return 0.55
    neutral = {"black", "white", "cream", "ivory", "grey", "gray", "navy", "beige", "brown", "denim"}
    if all(color in neutral for color in colors):
        return 0.95
    if any(color in neutral for color in colors):
        return 0.85
    return 0.7 if len(set(colors)) <= 2 else 0.45


def _occasion_compatibility(items: Sequence[WardrobeItem], occasion: str) -> float:
    target = occasion.lower()
    categories = {_category(item) for item in items}
    if target in {"wedding guest", "date night"} and ("dress" in categories or "outerwear" in categories):
        return 0.9
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
    if components["wardrobe_availability"] == 1.0:
        reasons.append(f"Every piece is available in the wardrobe ({len(items)} selected).")
    return tuple(reasons)
