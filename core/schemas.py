from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WardrobeItem:
    name: str
    item_id: str = ""
    category: str = "Uncategorized"
    color: Optional[str] = None
    image_name: Optional[str] = None
    image_path: Optional[str] = None
    image_hash: Optional[str] = None
    embedding: Optional[List[float]] = None
    confidence: Optional[float] = None
    pattern: Optional[str] = None
    mask_path: Optional[str] = None
    model_status: Optional[str] = None
    subcategory: Optional[str] = None
    market_category: str = "Unknown"
    style: Optional[str] = None
    fit: Optional[str] = None
    occasion: Optional[str] = None
    last_worn: Optional[str] = None
    times_worn: int = 0
    secondary_color: Optional[str] = None
    date_added: Optional[str] = None
    suitable_occasions: List[str] = field(default_factory=list)
    extracted_features: dict = field(default_factory=dict)
    classification_uncertain: bool = False


@dataclass
class UserProfile:
    gender: Optional[str] = None
    face_shape: Optional[str] = None
    skin_tone: Optional[str] = None
    undertone: Optional[str] = None
    body_shape: Optional[str] = None
    wardrobe_items: List[WardrobeItem] = field(default_factory=list)
    preferred_colors: List[str] = field(default_factory=list)
    style_preferences: List[str] = field(default_factory=list)
    occasion: str = "Everyday"
    profile_image_path: Optional[str] = None
    analysis_date: Optional[str] = None
    analysis_timestamp: Optional[str] = None
    extracted_features: dict = field(default_factory=dict)

    @property
    def analysis_count(self) -> int:
        return sum(value is not None for value in (self.face_shape, self.skin_tone, self.body_shape))
