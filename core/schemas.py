from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WardrobeItem:
    name: str
    item_id: str = ""
    category: str = "Uncategorized"
    color: Optional[str] = None
    image_name: Optional[str] = None
    embedding: Optional[List[float]] = None
    confidence: Optional[float] = None
    pattern: Optional[str] = None
    mask_path: Optional[str] = None
    model_status: Optional[str] = None


@dataclass
class UserProfile:
    face_shape: Optional[str] = None
    skin_tone: Optional[str] = None
    body_shape: Optional[str] = None
    wardrobe_items: List[WardrobeItem] = field(default_factory=list)
    preferred_colors: List[str] = field(default_factory=list)
    style_preferences: List[str] = field(default_factory=list)
    occasion: str = "Everyday"

    @property
    def analysis_count(self) -> int:
        return sum(value is not None for value in (self.face_shape, self.skin_tone, self.body_shape))
