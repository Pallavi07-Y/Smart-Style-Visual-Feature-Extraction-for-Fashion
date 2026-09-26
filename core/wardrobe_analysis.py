from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

GENERAL_MODEL_ID = "openai/clip-vit-base-patch32"
FASHION_MODEL_ID = "patrickjohncyh/fashion-clip"
DEVICE = "cpu"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERAL_PROMPTS = {
    "Clothing item": "a product photo of a single clothing item or garment",
    "Fashion item": "a fashion product photo where the item of clothing is the main subject",
    "Garment": "a photo of a garment of clothing",
    "Visible outfit": "a photo of a person wearing clearly visible clothes, with the outfit as the focus",
    "Fashion on person": "a fashion photo showing what a person is wearing",
    "Clothes on mannequin": "a product photo of clothes displayed on a mannequin",
    "Clothes laid out": "a product photo of clothes laid out for sale",
    "Landscape": "a photo of a landscape",
    "Farm": "a photo of a farm, field, or farmland",
    "Nature": "a photo of nature, plants, or trees",
    "Building": "a photo of a building or architecture",
    "Animal": "a photo of an animal",
    "Food": "a photo of food",
    "Vehicle": "a photo of a vehicle",
    "Room": "a photo of a room or interior",
    "Object": "a photo of a random everyday object",
    "Person without clothing focus": "a portrait or person photo where clothing is not visible or is not the subject",
}
POSITIVE_GATE_LABELS = (
    "Clothing item", "Fashion item", "Garment", "Visible outfit",
    "Fashion on person", "Clothes on mannequin", "Clothes laid out",
)
NEGATIVE_GATE_LABELS = (
    "Landscape", "Farm", "Nature", "Building", "Animal", "Food",
    "Vehicle", "Room", "Object", "Person without clothing focus",
)

FASHION_PROMPTS = {
    "Shirt": "a product photo of a shirt",
    "T-shirt": "a product photo of a t-shirt",
    "Top": "a product photo of a top",
    "Kurti": "a product photo of a kurti",
    "Dress": "a product photo of a dress",
    "Saree": "a product photo of a saree",
    "Skirt": "a product photo of a skirt",
    "Jeans": "a product photo of jeans",
    "Trousers": "a product photo of trousers",
    "Shorts": "a product photo of shorts",
    "Jacket": "a product photo of a jacket",
    "Coat": "a product photo of a coat",
    "Sweater": "a product photo of a sweater",
    "Hoodie": "a product photo of a hoodie",
    "Blouse": "a product photo of a blouse",
    "Shoes": "a product photo of shoes",
    "Bag": "a product photo of a bag",
    "Accessory": "a product photo of a fashion accessory",
    "Other clothing item": "a product photo of another clothing item",
}
STYLE_PROMPTS = {
    "Casual": "a photo of clothing in casual everyday style",
    "Formal": "a photo of clothing in formal office style",
    "Ethnic": "a photo of clothing in ethnic style",
    "Traditional": "a photo of clothing in traditional style",
    "Party": "a photo of clothing in party style",
    "Streetwear": "a photo of clothing in streetwear style",
    "Smart Casual": "a photo of clothing in smart casual style",
    "Minimal": "a photo of clothing in minimal style",
    "Sportswear": "a photo of clothing in sportswear style",
}
MARKET_PROMPTS = {
    "Women's": "a fashion garment specifically designed for women",
    "Men's": "a fashion garment specifically designed for men",
    "Unisex": "a unisex fashion garment designed for all genders",
    "Unknown": "a fashion garment with no clear gender-specific market",
}

GATE_SCORE_THRESHOLD = 0.25
GATE_MARGIN_THRESHOLD = 0.08
CATEGORY_SCORE_THRESHOLD = 0.32
CATEGORY_MARGIN_THRESHOLD = 0.08
STYLE_SCORE_THRESHOLD = 0.30
STYLE_MARGIN_THRESHOLD = 0.06
MARKET_SCORE_THRESHOLD = 0.65
MARKET_MARGIN_THRESHOLD = 0.20


class NotClothingImageError(ValueError):
    """Raised when the general vision gate does not find sufficient clothing evidence."""

    def __init__(self, analysis_details: dict[str, Any]):
        super().__init__("No clothing item was detected with sufficient confidence.")
        self.analysis_details = analysis_details


@st.cache_resource(show_spinner="Loading general vision model for first use...")
def _load_general_model():
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(GENERAL_MODEL_ID)
    model = AutoModel.from_pretrained(GENERAL_MODEL_ID, use_safetensors=True).to(torch.device(DEVICE)).eval()
    return processor, model


@st.cache_resource(show_spinner="Loading FashionCLIP for first use...")
def _load_fashion_model():
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(FASHION_MODEL_ID)
    model = AutoModel.from_pretrained(FASHION_MODEL_ID, use_safetensors=True).to(torch.device(DEVICE)).eval()
    return processor, model


@st.cache_resource(show_spinner="Loading lightweight clothing-region detector...")
def _load_person_detector():
    from ultralytics import YOLO

    return YOLO(str(PROJECT_ROOT / "models" / "yolo11n.pt"))


def _prompt_scores(image: Image.Image, prompts: dict[str, str], model_loader) -> dict[str, float]:
    import torch

    processor, model = model_loader()
    inputs = processor(text=list(prompts.values()), images=image, return_tensors="pt", padding=True)
    with torch.inference_mode():
        probabilities = model(**inputs).logits_per_image[0].float().softmax(dim=0).tolist()
    return {label: float(score) for label, score in zip(prompts, probabilities)}


def _center_crop(image: Image.Image, horizontal: float = 0.12, vertical: float = 0.08) -> Image.Image:
    width, height = image.size
    return image.crop((
        int(width * horizontal),
        int(height * vertical),
        max(1, int(width * (1 - horizontal))),
        max(1, int(height * (1 - vertical))),
    ))


def _general_gate(image: Image.Image) -> tuple[bool, dict[str, Any]]:
    views = (image, _center_crop(image, horizontal=0.15, vertical=0.10))
    per_view = [_prompt_scores(view, GENERAL_PROMPTS, _load_general_model) for view in views]
    scores = {
        label: sum(view_scores[label] for view_scores in per_view) / len(per_view)
        for label in GENERAL_PROMPTS
    }
    clothing_score = sum(scores[label] for label in POSITIVE_GATE_LABELS)
    negative_label = max(NEGATIVE_GATE_LABELS, key=lambda label: scores[label])
    negative_score = scores[negative_label]
    margin = clothing_score - negative_score
    accepted = clothing_score >= GATE_SCORE_THRESHOLD and margin >= GATE_MARGIN_THRESHOLD
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    details = {
        "model": GENERAL_MODEL_ID,
        "device": DEVICE,
        "prediction": "Clothing" if accepted else "Non-clothing / insufficient clothing evidence",
        "decision": "accepted" if accepted else "rejected; image was not saved",
        "candidate_scores": {label: round(score, 4) for label, score in scores.items()},
        "top_predictions": [(label, round(score, 4)) for label, score in ranked[:6]],
        "clothing_evidence_score": round(clothing_score, 4),
        "strongest_non_clothing_label": negative_label,
        "strongest_non_clothing_score": round(negative_score, 4),
        "clothing_minus_non_clothing_margin": round(margin, 4),
        "thresholds": {
            "minimum_clothing_evidence": GATE_SCORE_THRESHOLD,
            "minimum_margin_over_non_clothing": GATE_MARGIN_THRESHOLD,
        },
        "score_note": "Softmax probabilities are relative to these candidate prompts, not calibrated probabilities.",
    }
    return accepted, details


def _margin(scores: dict[str, float]) -> float:
    values = sorted(scores.values(), reverse=True)
    return values[0] - values[1] if len(values) > 1 else (values[0] if values else 0.0)


def _broad_category(label: str) -> str:
    if label in {"Shirt", "T-shirt", "Top", "Kurti", "Blouse"}:
        return "Top"
    if label in {"Jeans", "Trousers", "Shorts", "Skirt"}:
        return "Bottom"
    if label in {"Jacket", "Coat", "Sweater", "Hoodie"}:
        return "Outerwear"
    if label in {"Saree", "Dress"}:
        return "Dress"
    if label == "Shoes":
        return "Shoes"
    if label in {"Bag", "Accessory"}:
        return "Accessory"
    return "Uncategorized"


def _style_result(image: Image.Image, category: str) -> tuple[str, float, dict[str, float]]:
    scores = _prompt_scores(image, STYLE_PROMPTS, _load_fashion_model)
    label = max(scores, key=scores.get)
    confidence = scores[label]
    margin = _margin(scores)
    if confidence >= STYLE_SCORE_THRESHOLD and margin >= STYLE_MARGIN_THRESHOLD:
        return label, confidence, scores
    deterministic_style = {"Kurti": "Ethnic", "Saree": "Traditional"}.get(category)
    if deterministic_style:
        return deterministic_style, confidence, scores
    return "Needs confirmation", confidence, scores


def _market_result(image: Image.Image) -> tuple[str, float, dict[str, float]]:
    scores = _prompt_scores(image, MARKET_PROMPTS, _load_fashion_model)
    label = max(scores, key=scores.get)
    confidence = scores[label]
    if confidence < MARKET_SCORE_THRESHOLD or _margin(scores) < MARKET_MARGIN_THRESHOLD:
        return "Unknown", confidence, scores
    return label, confidence, scores


def _clothing_region(image: Image.Image, garment: str) -> tuple[Image.Image, str]:
    detector = _load_person_detector()
    detection = detector.predict(source=image, device=DEVICE, classes=[0], conf=0.20, verbose=False)[0]
    width, height = image.size
    boxes = [box for box in detection.boxes if int(box.cls[0]) == 0]
    if boxes:
        person = max(boxes, key=lambda box: float(box.conf[0]))
        x1, y1, x2, y2 = [float(value) for value in person.xyxy[0].tolist()]
        box_width, box_height = x2 - x1, y2 - y1
        vertical_regions = {
            "Shirt": (0.18, 0.64), "T-shirt": (0.18, 0.64), "Top": (0.18, 0.64),
            "Kurti": (0.18, 0.78), "Blouse": (0.18, 0.64),
            "Jeans": (0.42, 0.98), "Trousers": (0.42, 0.98),
            "Shorts": (0.42, 0.78), "Skirt": (0.35, 0.90),
            "Shoes": (0.78, 1.0), "Dress": (0.16, 0.96), "Saree": (0.12, 0.98),
        }
        top, bottom = vertical_regions.get(garment, (0.18, 0.85))
        crop = image.crop((
            max(0, int(x1 + box_width * 0.08)),
            max(0, int(y1 + box_height * top)),
            min(width, int(x2 - box_width * 0.08)),
            min(height, int(y1 + box_height * bottom)),
        ))
        return crop, f"YOLO person box; category-specific clothing-area crop ({garment}); approximate, not segmentation"

    crop = _center_crop(image, horizontal=0.18, vertical=0.12)
    return crop, "Centered object crop; approximate because no person box was detected"


def _color_name(rgb: np.ndarray) -> str:
    red, green, blue = [float(value) for value in rgb]
    if max(red, green, blue) < 55:
        return "Black"
    if min(red, green, blue) > 205:
        return "White"
    if max(red, green, blue) - min(red, green, blue) < 25:
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


def _color_features(region: Image.Image) -> tuple[str, str | None, dict[str, Any]]:
    import cv2
    from sklearn.cluster import KMeans

    region = ImageOps.contain(region, (128, 128)).convert("RGB")
    rgb = np.asarray(region, dtype=np.uint8)
    height, width = rgb.shape[:2]
    segmentation = "GrabCut foreground pixels inside detected clothing-area crop"
    try:
        mask = np.zeros((height, width), dtype=np.uint8)
        margin_x, margin_y = max(1, width // 24), max(1, height // 24)
        rect = (margin_x, margin_y, max(1, width - 2 * margin_x), max(1, height - 2 * margin_y))
        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect,
            background_model, foreground_model, 4, cv2.GC_INIT_WITH_RECT,
        )
        foreground_mask = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
        foreground_fraction = float(foreground_mask.mean())
        if 0.08 <= foreground_fraction <= 0.98:
            pixels = rgb[foreground_mask].astype(np.float32)
        else:
            pixels = rgb.astype(np.float32).reshape(-1, 3)
            segmentation = "Detected clothing-area crop only; GrabCut mask unusable, approximate"
    except cv2.error:
        pixels = rgb.astype(np.float32).reshape(-1, 3)
        segmentation = "Detected clothing-area crop only; GrabCut failed, approximate"
    if not len(pixels):
        return "Unable to determine", None, {"scope": "Clothing region unavailable"}
    unique = np.unique(pixels, axis=0)
    if len(unique) < 2:
        primary = _color_name(pixels[0])
        return primary, None, {"scope": segmentation, "color_proportions": {primary: 1.0}}

    cluster_count = min(3, len(unique))
    model = KMeans(n_clusters=cluster_count, n_init=5, random_state=42).fit(pixels)
    counts = np.bincount(model.labels_)
    order = np.argsort(counts)[::-1]
    names = [_color_name(model.cluster_centers_[index]) for index in order]
    proportions: dict[str, float] = {}
    for index, cluster_index in enumerate(order):
        name = names[index]
        proportions[name] = proportions.get(name, 0.0) + float(counts[cluster_index] / len(pixels))
    proportions = {name: round(value, 3) for name, value in proportions.items()}
    primary = names[0]
    secondary = next((name for name in names[1:] if name != primary and proportions[name] >= 0.20), None)
    return primary, secondary, {
        "scope": segmentation,
        "color_proportions": proportions,
    }


def _suitable_occasions(category: str, garment: str, style: str) -> list[str]:
    if style == "Formal":
        return ["Interview", "Presentation", "Office", "Formal Event"]
    if style == "Party":
        return ["Party", "Dinner", "Evening Event"]
    if style in {"Ethnic", "Traditional"} or garment in {"Kurti", "Saree"}:
        return ["College", "Casual", "Festive Casual", "Family Gathering"]
    if style == "Sportswear":
        return ["Workout", "Outdoor", "Travel"]
    if style == "Smart Casual":
        return ["Office", "College", "Casual", "Presentation"]
    if category in {"Shoes", "Accessory"}:
        return ["College", "Casual", "Travel"]
    return ["College", "Casual", "Travel", "Outdoor"]


def analyze_wardrobe_image(image_bytes: bytes, filename: str = "") -> dict[str, Any]:
    """Run general CLIP gating, then FashionCLIP analysis on a verified clothing image."""
    if not image_bytes:
        raise ValueError("The wardrobe image is empty.")
    try:
        image = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error

    accepted, gate_details = _general_gate(image)
    if not accepted:
        raise NotClothingImageError({
            "general_vision_classification": gate_details,
            "fashion_classification": {"decision": "not run; general gate rejected this image"},
            "final_result": "No clothing item detected; not saved",
        })

    fashion_scores = _prompt_scores(image, FASHION_PROMPTS, _load_fashion_model)
    garment = max(fashion_scores, key=fashion_scores.get)
    garment_confidence = fashion_scores[garment]
    garment_margin = _margin(fashion_scores)
    garment_certain = (
        garment != "Other clothing item"
        and garment_confidence >= CATEGORY_SCORE_THRESHOLD
        and garment_margin >= CATEGORY_MARGIN_THRESHOLD
    )
    displayed_garment = garment if garment_certain else "Unable to determine"
    category = _broad_category(garment) if garment_certain else "Uncategorized"

    if garment_certain:
        style, style_confidence, style_scores = _style_result(image, garment)
        market_category, market_confidence, market_scores = _market_result(image)
        region, region_method = _clothing_region(image, garment)
        primary_color, secondary_color, color_details = _color_features(region)
    else:
        style, style_confidence, style_scores = "Needs confirmation", 0.0, {}
        market_category, market_confidence, market_scores = "Unknown", 0.0, {}
        primary_color, secondary_color = "Unable to determine", None
        region_method = "not run because clothing type is uncertain"
        color_details = {"scope": region_method}

    style_margin = _margin(style_scores)
    style_certain = style in STYLE_PROMPTS and style_confidence >= STYLE_SCORE_THRESHOLD and style_margin >= STYLE_MARGIN_THRESHOLD
    classification_uncertain = not garment_certain or not style_certain or style == "Needs confirmation"
    occasions = _suitable_occasions(category, garment, style) if garment_certain else []

    details = {
        "general_vision_classification": gate_details,
        "fashion_classification": {
            "model": FASHION_MODEL_ID,
            "device": DEVICE,
            "candidate_scores": {label: round(score, 4) for label, score in fashion_scores.items()},
            "top_predictions": [(label, round(score, 4)) for label, score in sorted(fashion_scores.items(), key=lambda item: item[1], reverse=True)[:6]],
            "predicted_type": garment,
            "displayed_type": displayed_garment,
            "confidence": round(garment_confidence, 4),
            "margin": round(garment_margin, 4),
            "thresholds": {"minimum_confidence": CATEGORY_SCORE_THRESHOLD, "minimum_margin": CATEGORY_MARGIN_THRESHOLD},
        },
        "style_classification": {
            "candidate_scores": {label: round(score, 4) for label, score in style_scores.items()},
            "result": style,
            "confidence": round(style_confidence, 4),
            "margin": round(style_margin, 4),
            "market_category": market_category,
            "market_candidate_scores": {label: round(score, 4) for label, score in market_scores.items()},
            "market_confidence": round(market_confidence, 4),
        },
        "clothing_region": {"method": region_method},
        "color_analysis": color_details,
        "occasion_mapping": {"category": category, "garment": displayed_garment, "style": style, "result": occasions},
        "final_result": {
            "decision": "accepted with confirmation required" if classification_uncertain else "accepted",
            "category": category,
            "clothing_type": displayed_garment,
            "market_category": market_category,
            "primary_color": primary_color,
            "secondary_color": secondary_color,
            "style": style,
            "suitable_occasions": occasions,
        },
    }
    return {
        "category": category,
        "subcategory": displayed_garment,
        "market_category": market_category,
        "confidence": round(garment_confidence, 4),
        "color": primary_color,
        "secondary_color": secondary_color,
        "style": style,
        "suitable_occasions": occasions,
        "classification_uncertain": classification_uncertain,
        "model_status": f"FashionCLIP CPU; {garment_confidence:.0%} relative category score; clothing-area color crop is approximate",
        "extracted_features": {
            "vision_models": {"gate": GENERAL_MODEL_ID, "fashion": FASHION_MODEL_ID, "device": DEVICE},
            "analysis_details": details,
        },
    }