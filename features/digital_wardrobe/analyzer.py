from dataclasses import dataclass
import io
import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)
DEFAULT_FASHIONCLIP_MODEL = "patrickjohncyh/fashion-clip"
DEFAULT_SAM2_MODEL = "facebook/sam2-hiera-small"


@dataclass(frozen=True)
class FashionEmbedding:
    embedding: list[float]
    model_name: str
    device: str


@dataclass(frozen=True)
class SimilarityMatch:
    item_id: str
    score: float
    metadata: Any = None


@dataclass(frozen=True)
class WardrobeTagResult:
    category: str
    color: str
    pattern: str
    mask_path: Optional[str]
    embedding: Optional[list[float]]
    confidence: float
    model_status: str


@dataclass(frozen=True)
class SegmentedClothingItem:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray
    mask_path: str
    crop_path: str
    crop_image: Any
    embedding: list[float]
    color: str
    pattern: str
    model_status: str


def resolve_device(device: str | None = None) -> str:
    """Choose the requested device, falling back to CPU when CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if device:
        if device.lower().startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable; falling back to CPU.")
            return "cpu"
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _to_pil_image(image: Any):
    from PIL import Image

    if isinstance(image, (bytes, bytearray, memoryview)):
        try:
            return Image.open(io.BytesIO(image)).convert("RGB")
        except Exception as error:
            raise ValueError("The clothing image is not readable.") from error
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("The clothing image is empty.")
        array = image
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Expected an RGB, RGBA, or grayscale image.")
        return Image.fromarray(array[:, :, :3]).convert("RGB")
    raise TypeError("Image must be bytes, a PIL image, or a NumPy array.")


def _load_fashionclip(model_name: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def _load_sam2(model_name: str, device: str):
    from transformers import Sam2Model, Sam2Processor

    processor = Sam2Processor.from_pretrained(model_name)
    model = Sam2Model.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def _feature_tensor(features: Any):
    """Normalize Transformers versions that return either tensors or output objects."""
    if isinstance(features, tuple):
        return features[0]
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


try:
    import streamlit as _streamlit
except ImportError:  # Allows library and unit-test use outside Streamlit.
    _streamlit = None

if _streamlit is not None:
    _load_fashionclip = _streamlit.cache_resource(show_spinner=False)(_load_fashionclip)
    _load_sam2 = _streamlit.cache_resource(show_spinner=False)(_load_sam2)


def segment_clothing_items(
    image: Any,
    detections: Iterable[Any],
    output_dir: str | Path = "data/wardrobe/masks",
    sam_model: str = DEFAULT_SAM2_MODEL,
    fashionclip_model: str = DEFAULT_FASHIONCLIP_MODEL,
    device: str | None = None,
) -> list[SegmentedClothingItem]:
    """Segment YOLO boxes with official SAM2 and embed each masked crop."""
    pil_image = _to_pil_image(image)
    detection_list = list(detections)
    if not detection_list:
        return []
    inference_device = resolve_device(device)
    try:
        processor, model = _load_sam2(sam_model, inference_device)
    except Exception as error:
        LOGGER.exception("SAM2 model loading failed for %s", sam_model)
        raise RuntimeError(f"SAM2 model loading failed: {error}") from error

    width, height = pil_image.size
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[SegmentedClothingItem] = []
    for index, detection in enumerate(detection_list):
        bbox = tuple(float(value) for value in getattr(detection, "bbox", getattr(detection, "box", ())))
        if len(bbox) != 4:
            raise ValueError("Each detection must provide a four-value pixel bounding box.")
        x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
        try:
            inputs = processor(images=pil_image, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
            inputs = {name: value.to(inference_device) for name, value in inputs.items() if hasattr(value, "to")}
            import torch

            with torch.inference_mode():
                outputs = model(**inputs, multimask_output=False)
            resized_sizes = torch.tensor(
                [[inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]]],
                device=inputs["pixel_values"].device,
            )
            processed_masks = processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                resized_sizes.cpu(),
            )
            mask = processed_masks[0][0][0].numpy().astype(bool)
        except Exception as error:
            LOGGER.exception("SAM2 segmentation failed for detection %s", index)
            raise RuntimeError(f"SAM2 segmentation failed for detection {index}: {error}") from error

        crop = _masked_crop(pil_image, mask, (x1, y1, x2, y2))
        crop_bytes = io.BytesIO()
        crop.save(crop_bytes, format="PNG")
        embedding = get_fashion_embedding(crop, fashionclip_model, inference_device).embedding
        stem = f"item_{index}"
        mask_file = output_path / f"{stem}_mask.png"
        crop_file = output_path / f"{stem}_crop.png"
        from PIL import Image

        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_file)
        crop.save(crop_file)
        label = str(getattr(detection, "label", getattr(detection, "category", "unknown")))
        confidence = float(getattr(detection, "confidence", 0.0))
        results.append(SegmentedClothingItem(
            label=label,
            confidence=confidence,
            bbox=(x1, y1, x2, y2),
            mask=mask,
            mask_path=str(mask_file),
            crop_path=str(crop_file),
            crop_image=crop,
            embedding=embedding,
            color=_dominant_color(crop),
            pattern=_pattern_hint(crop),
            model_status=f"SAM2 + FashionCLIP ready on {inference_device}",
        ))
    return results


def _clamp_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left, top = max(0, min(int(round(x1)), width - 1)), max(0, min(int(round(y1)), height - 1))
    right, bottom = max(left + 1, min(int(round(x2)), width)), max(top + 1, min(int(round(y2)), height))
    return left, top, right, bottom


def _masked_crop(image: Any, mask: np.ndarray, bbox: tuple[int, int, int, int]):
    from PIL import Image

    x1, y1, x2, y2 = bbox
    crop = np.asarray(image).copy()
    crop[~mask] = 255
    return Image.fromarray(crop[y1:y2, x1:x2]).convert("RGB")


def _dominant_color(image: Any) -> str:
    pixels = np.asarray(image).reshape(-1, 3).mean(axis=0)
    names = {"black": 45, "white": 220, "gray": 150, "red": (180, 60, 60), "blue": (60, 100, 180), "green": (70, 150, 80), "yellow": (190, 170, 60)}
    if pixels.mean() < 45:
        return "black"
    if pixels.mean() > 220:
        return "white"
    if np.max(pixels) - np.min(pixels) < 25:
        return "gray"
    return min((name for name in names if isinstance(names[name], tuple)), key=lambda name: float(np.linalg.norm(pixels - np.asarray(names[name]))))


def _pattern_hint(image: Any) -> str:
    values = np.asarray(image).astype(np.float32)
    return "textured" if float(values.std()) > 55 else "solid"


def get_fashion_embedding(
    image: Any,
    model_name: str = DEFAULT_FASHIONCLIP_MODEL,
    device: str | None = None,
) -> FashionEmbedding:
    """Generate a normalized FashionCLIP image embedding for a clothing crop."""
    pil_image = _to_pil_image(image)
    inference_device = resolve_device(device)
    try:
        processor, model = _load_fashionclip(model_name, inference_device)
        import torch

        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {name: value.to(inference_device) for name, value in inputs.items()}
        with torch.inference_mode():
            embedding = _feature_tensor(model.get_image_features(**inputs))
            embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        values = embedding[0].detach().float().cpu().tolist()
    except Exception as error:
        LOGGER.exception("FashionCLIP embedding failed for model %s", model_name)
        raise RuntimeError(f"FashionCLIP embedding failed: {error}") from error
    return FashionEmbedding(embedding=values, model_name=model_name, device=inference_device)


def get_fashion_text_similarity(
    image: Any,
    labels: Sequence[str],
    model_name: str = DEFAULT_FASHIONCLIP_MODEL,
    device: str | None = None,
) -> dict[str, float]:
    """Score caller-provided text labels without inventing or assigning labels."""
    if not labels:
        return {}
    pil_image = _to_pil_image(image)
    inference_device = resolve_device(device)
    try:
        processor, model = _load_fashionclip(model_name, inference_device)
        import torch

        inputs = processor(text=list(labels), images=pil_image, return_tensors="pt", padding=True)
        inputs = {name: value.to(inference_device) for name, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits_per_image[0]
            scores = torch.softmax(logits, dim=0).detach().float().cpu().tolist()
    except Exception as error:
        LOGGER.exception("FashionCLIP text similarity failed for model %s", model_name)
        raise RuntimeError(f"FashionCLIP text similarity failed: {error}") from error
    return {label: round(float(score), 6) for label, score in zip(labels, scores)}


def find_similar_items(
    query_embedding: Sequence[float],
    wardrobe_embeddings: Iterable[Any],
    top_k: int = 5,
) -> list[SimilarityMatch]:
    """Return wardrobe entries ranked by cosine similarity to a query vector."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim != 1 or query.size == 0:
        raise ValueError("query_embedding must be a non-empty one-dimensional vector.")
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise ValueError("query_embedding must not be a zero vector.")

    matches: list[SimilarityMatch] = []
    for index, entry in enumerate(wardrobe_embeddings):
        item_id, vector, metadata = _embedding_entry(entry, index)
        candidate = np.asarray(vector, dtype=np.float32)
        if candidate.shape != query.shape:
            raise ValueError(f"Embedding dimension mismatch for wardrobe item {item_id}.")
        candidate_norm = np.linalg.norm(candidate)
        if candidate_norm == 0:
            continue
        score = float(np.dot(query, candidate) / (query_norm * candidate_norm))
        matches.append(SimilarityMatch(item_id=item_id, score=round(score, 6), metadata=metadata))
    return sorted(matches, key=lambda match: match.score, reverse=True)[:top_k]


def _embedding_entry(entry: Any, index: int) -> tuple[str, Sequence[float], Any]:
    if isinstance(entry, dict):
        vector = entry.get("embedding")
        item_id = str(entry.get("item_id", index))
        return item_id, vector, entry
    if hasattr(entry, "embedding"):
        return str(getattr(entry, "item_id", index)), entry.embedding, entry
    if isinstance(entry, tuple) and len(entry) == 2:
        return str(entry[0]), entry[1], None
    return str(index), entry, None


def segment_and_tag_clothing(
    image_bytes: bytes,
    sam_checkpoint: str | Path | None = None,
    fashionclip_model: str = "patrickjohncyh/fashion-clip",
    output_dir: str | Path = "data/wardrobe/masks",
) -> WardrobeTagResult:
    """Segment an item with SAM 2 and tag it with FashionCLIP.

    Heavy model imports are intentionally lazy so the wardrobe editor still works
    while checkpoints are being installed.
    """
    if not image_bytes:
        raise ValueError("The clothing image is empty.")
    if sam_checkpoint and Path(sam_checkpoint).exists():
        raise ValueError("Local SAM2 .pt checkpoints require a matching native SAM2 configuration; use the official Transformers SAM2 model ID.")
    raise ValueError("Provide YOLO detections to segment_clothing_items; single-image tagging cannot infer a clothing box safely.")
