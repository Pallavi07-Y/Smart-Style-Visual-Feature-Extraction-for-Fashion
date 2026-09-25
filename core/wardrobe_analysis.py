from pathlib import Path
from typing import Any


def analyze_wardrobe_image(image_bytes: bytes, filename: str) -> dict[str, Any]:
    """Perform lightweight, local wardrobe tagging suitable for CPU-only use.

    Filename clues are used for clothing type/style and image pixels for color;
    every value remains editable because a single photo cannot guarantee exact
    garment semantics.
    """
    if not image_bytes:
        raise ValueError("The wardrobe image is empty.")
    from PIL import Image
    import io
    import numpy as np

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixels = np.asarray(image.resize((48, 48)), dtype=np.float32).reshape(-1, 3)
    primary_color, secondary_color, proportions = _dominant_colors(pixels)
    text = Path(filename).stem.lower().replace("_", " ")
    category = _category(text)
    classification_uncertain = not any(token in text for token in ("kurti", "kurta", "saree", "shirt", "tshirt", "tee", "jean", "trouser", "pant", "dress", "shoe", "sneaker", "flat", "bag", "jacket", "blazer", "skirt"))
    style = _style(text, category)
    occasions = _occasions(category, style)
    return {
        "category": category,
        "color": primary_color,
        "secondary_color": secondary_color,
        "style": style,
        "suitable_occasions": occasions,
        "classification_uncertain": classification_uncertain,
        "extracted_features": {"dominant_rgb": [round(float(value), 2) for value in pixels.mean(axis=0)], "color_proportions": proportions, "preprocessing": "RGB conversion and 48x48 resize"},
        "model_status": "Local CPU color and filename analysis; edit to confirm",
    }


def _category(text: str) -> str:
    if any(word in text for word in ("shoe", "sneaker", "flat", "boot", "sandal")):
        return "Shoes"
    if any(word in text for word in ("bag", "purse", "handbag")):
        return "Bag"
    if any(word in text for word in ("accessory", "scarf", "belt", "jewelry", "jewel")):
        return "Accessory"
    if any(word in text for word in ("saree", "sari")):
        return "Saree"
    if any(word in text for word in ("kurti", "kurta", "anarkali")):
        return "Kurti"
    if any(word in text for word in ("jean", "trouser", "pant", "skirt", "short")):
        return "Bottom"
    if any(word in text for word in ("dress", "lehenga", "sharara")):
        return "Dress"
    if any(word in text for word in ("jacket", "blazer", "coat", "cardigan")):
        return "Outerwear"
    if "tshirt" in text or "t-shirt" in text or "tee" in text:
        return "T-shirt"
    return "Shirt"


def _style(text: str, category: str) -> str:
    if any(word in text for word in ("formal", "blazer", "suit")):
        return "Formal"
    if any(word in text for word in ("ethnic", "kurti", "kurta", "saree", "anarkali")):
        return "Ethnic"
    if any(word in text for word in ("party", "sequin", "lehenga", "sharara")):
        return "Party"
    if any(word in text for word in ("street", "oversized", "sneaker")):
        return "Streetwear"
    return "Traditional" if category in {"Saree", "Kurti"} else "Casual"


def _occasions(category: str, style: str) -> list[str]:
    if style == "Formal":
        return ["Formal", "Interview", "Presentation"]
    if style in {"Ethnic", "Traditional"}:
        return ["College", "Traditional", "Wedding", "Family Gathering"]
    if style == "Party":
        return ["Party", "Wedding", "Dinner"]
    if category in {"Shoes", "Bag", "Accessory"}:
        return ["College", "Casual", "Travel"]
    return ["College", "Casual", "Travel", "Outdoor"]


def _color_name(rgb) -> str:
    red, green, blue = rgb
    if max(rgb) < 55:
        return "Black"
    if min(rgb) > 205:
        return "White"
    if max(rgb) - min(rgb) < 25:
        return "Grey"
    if red > green * 1.35 and red > blue * 1.25:
        return "Red"
    if green > red * 1.2 and green > blue * 1.15:
        return "Green"
    if blue > red * 1.2 and blue > green * 1.05:
        return "Blue"
    if red > 145 and green > 90 and blue < 100:
        return "Orange"
    if red > 150 and blue > 120 and green < 130:
        return "Pink"
    return "Beige"


def _dominant_colors(pixels):
    import numpy as np
    from sklearn.cluster import KMeans

    if len(np.unique(pixels, axis=0)) < 2:
        name = _color_name(pixels[0])
        return name, None, {name: 1.0}
    model = KMeans(n_clusters=2, n_init=5, random_state=42).fit(pixels)
    counts = np.bincount(model.labels_)
    order = np.argsort(counts)[::-1]
    names = [_color_name(model.cluster_centers_[index]) for index in order]
    proportions = {names[index]: round(float(counts[order[index]] / len(pixels)), 3) for index in range(2)}
    return names[0], names[1] if names[1] != names[0] else None, proportions