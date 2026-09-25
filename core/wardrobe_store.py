import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from core.schemas import UserProfile, WardrobeItem


class WardrobeStore:
    """Small JSON-backed repository for one user's wardrobe metadata."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[WardrobeItem]:
        if not self.path.exists():
            return []
        try:
            records = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [WardrobeItem(**record) for record in records]

    def save(self, items: Iterable[WardrobeItem]) -> None:
        payload = [asdict(item) for item in items]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, item: WardrobeItem) -> WardrobeItem:
        if not item.item_id:
            item.item_id = uuid4().hex
        items = self.load()
        items.append(item)
        self.save(items)
        return item

    def save_image(self, item_id: str, filename: str, content: bytes, directory: str | Path | None = None) -> Path:
        """Persist the original upload so recommendations can render real wardrobe images."""
        target_dir = Path(directory) if directory else self.path.parent / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
        destination = target_dir / f"{item_id}_{safe_name}"
        destination.write_bytes(content)
        return destination

    def update(self, item: WardrobeItem) -> None:
        items = self.load()
        for index, existing in enumerate(items):
            if existing.item_id == item.item_id:
                items[index] = item
                self.save(items)
                return
        raise KeyError(f"Wardrobe item not found: {item.item_id}")

    def delete(self, item_id: str) -> None:
        self.save(item for item in self.load() if item.item_id != item_id)

    @property
    def weekly_plan_path(self) -> Path:
        return self.path.parent / "weekly_plan.json"

    def save_weekly_plan(self, plan: dict) -> None:
        self.weekly_plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def load_weekly_plan(self) -> dict:
        if not self.weekly_plan_path.exists():
            return {}

    @property
    def profile_path(self) -> Path:
        return self.path.parent / "profile.json"

    def save_profile(self, profile: UserProfile) -> None:
        record = {
            "gender": profile.gender,
            "face_shape": profile.face_shape,
            "skin_tone": profile.skin_tone,
            "undertone": profile.undertone,
            "body_shape": profile.body_shape,
            "preferred_colors": profile.preferred_colors,
            "style_preferences": profile.style_preferences,
            "occasion": profile.occasion,
            "profile_image_path": profile.profile_image_path,
            "analysis_date": profile.analysis_date,
            "extracted_features": profile.extracted_features,
        }
        self.profile_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def load_profile(self) -> UserProfile:
        if not self.profile_path.exists():
            return UserProfile()
        try:
            return UserProfile(**json.loads(self.profile_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return UserProfile()
        try:
            return json.loads(self.weekly_plan_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
