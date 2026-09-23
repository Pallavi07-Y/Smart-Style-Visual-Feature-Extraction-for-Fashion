from dataclasses import dataclass
from typing import Iterable, Sequence

from core.schemas import UserProfile, WardrobeItem
from features.recommendation import OutfitCandidate, recommend_outfits

DEFAULT_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class PlannedDay:
    day: str
    occasion: str
    outfit: OutfitCandidate | None
    reason: str


@dataclass(frozen=True)
class WeeklyPlan:
    days: tuple[PlannedDay, ...]


def generate_weekly_plan(
    profile: UserProfile,
    wardrobe_items: Iterable[WardrobeItem] | None = None,
    occasions: Sequence[str] | None = None,
    days: Sequence[str] = DEFAULT_DAYS,
) -> WeeklyPlan:
    """Build a seven-day plan from the same recommendation scorer used elsewhere."""
    items = list(wardrobe_items if wardrobe_items is not None else profile.wardrobe_items)
    selected_occasions = list(occasions or [profile.occasion or "Everyday"] * len(days))
    if len(selected_occasions) != len(days):
        raise ValueError("occasions must have one value per day.")
    planned: list[PlannedDay] = []
    recent_ids: set[str] = set()
    for day, occasion in zip(days, selected_occasions):
        candidates = recommend_outfits(profile, items, occasion=occasion, top_k=10)
        candidate = _prefer_non_repeating(candidates, recent_ids) or (candidates[0] if candidates else None)
        if candidate:
            recent_ids.update(item.item_id for item in candidate.items)
            reason = " ".join(candidate.reasons[:2])
        else:
            reason = "Not enough categorized wardrobe items to build an outfit."
        planned.append(PlannedDay(day=day, occasion=occasion, outfit=candidate, reason=reason))
    return WeeklyPlan(days=tuple(planned))


def regenerate_day(
    plan: WeeklyPlan,
    day_index: int,
    profile: UserProfile,
    wardrobe_items: Iterable[WardrobeItem] | None = None,
) -> WeeklyPlan:
    """Regenerate one day while excluding items used by the other planned days when possible."""
    if not 0 <= day_index < len(plan.days):
        raise IndexError("day_index is outside the weekly plan.")
    items = list(wardrobe_items if wardrobe_items is not None else profile.wardrobe_items)
    current = plan.days[day_index]
    used_elsewhere = {
        item.item_id
        for index, day in enumerate(plan.days)
        if index != day_index and day.outfit
        for item in day.outfit.items
    }
    alternatives = recommend_outfits(profile, items, occasion=current.occasion, top_k=20)
    replacement = next((candidate for candidate in alternatives if not any(item.item_id in used_elsewhere for item in candidate.items)), None)
    if replacement is None:
        replacement = next((candidate for candidate in alternatives if candidate.items != (current.outfit.items if current.outfit else ())), None)
    updated = list(plan.days)
    updated[day_index] = PlannedDay(
        day=current.day,
        occasion=current.occasion,
        outfit=replacement,
        reason=" ".join(replacement.reasons[:2]) if replacement else "Not enough wardrobe items to regenerate this day.",
    )
    return WeeklyPlan(days=tuple(updated))


def _prefer_non_repeating(candidates: Sequence[OutfitCandidate], recent_ids: set[str]) -> OutfitCandidate | None:
    for candidate in candidates:
        if not any(item.item_id in recent_ids for item in candidate.items):
            return candidate
    return None
