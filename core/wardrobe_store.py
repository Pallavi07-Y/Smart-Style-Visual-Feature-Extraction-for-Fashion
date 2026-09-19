import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from core.schemas import WardrobeItem


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
