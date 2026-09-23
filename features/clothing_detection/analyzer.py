from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

# These aliases are deliberately applied only when the loaded checkpoint exposes
# the source label. Unknown labels remain unchanged instead of being guessed.
_CATEGORY_ALIASES = {
    "t-shirt": "t-shirt",
    "tshirt": "t-shirt",
    "shirt": "shirt",
    "blouse": "blouse",
    "jacket": "jacket",
    "coat": "coat",
    "dress": "dress",
    "skirt": "skirt",
    "trousers": "trousers",
    "pants": "trousers",
    "jeans": "jeans",
    "shorts": "shorts",
    "shoes": "shoes",
    "shoe": "shoes",
    "handbag": "handbag",
    "accessory": "accessory",
}
_FASHION_LABELS = {
    "t-shirt",
    "tshirt",
    "shirt",
    "blouse",
    "jacket",
    "coat",
    "dress",
    "skirt",
    "trousers",
    "pants",
    "jeans",
    "shorts",
    "shoes",
    "shoe",
}


@dataclass(frozen=True)
class ClothingDetection:
    label: str
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    normalized_bbox: tuple[float, float, float, float]
    category: str
    embedding: Optional[list[float]] = None

    @property
    def box(self) -> tuple[float, float, float, float]:
        """Backward-compatible alias for the pixel bounding box."""
        return self.bbox


@dataclass(frozen=True)
class ClothingDetectionResult:
    detections: list[ClothingDetection]
    annotated_image: Any
    model_status: str
    model_classes: tuple[str, ...]
    device: str
    confidence_threshold: float


def _load_model(checkpoint: str):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("Install ultralytics to enable YOLO clothing detection.") from error
    return YOLO(checkpoint)


try:
    import streamlit as _streamlit
except ImportError:  # Allows analyzer unit tests outside a Streamlit process.
    _streamlit = None

if _streamlit is not None:
    _load_model = _streamlit.cache_resource(show_spinner=False)(_load_model)


def resolve_device(device: str | None = None) -> str:
    """Return the requested device, or CUDA when available with CPU fallback."""
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


def _to_rgb_array(image: Any):
    import io

    import numpy as np

    if isinstance(image, (bytes, bytearray, memoryview)):
        try:
            from PIL import Image

            image = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
        except Exception as error:
            raise ValueError("The uploaded image is not readable.") from error
    elif hasattr(image, "convert"):
        image = np.asarray(image.convert("RGB"))
    elif not isinstance(image, np.ndarray):
        raise TypeError("Image must be bytes, a PIL image, or a NumPy array.")

    if image.size == 0:
        raise ValueError("The clothing image is empty.")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB, RGBA, or grayscale image.")
    return np.ascontiguousarray(image)


def _class_name(names: Any, class_id: int) -> str | None:
    if isinstance(names, dict):
        return str(names[class_id]) if class_id in names else None
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return None


def _category_for(label: str) -> str:
    normalized = label.strip().lower()
    return _CATEGORY_ALIASES.get(normalized, label)


def detect_clothing(
    image: Any,
    yolo_checkpoint: str | Path = "models/yolo11n.pt",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str | None = None,
) -> ClothingDetectionResult:
    """Run YOLO inference on bytes, a PIL image, or a NumPy image array."""
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    checkpoint = Path(yolo_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint}")

    rgb_image = _to_rgb_array(image)
    inference_device = resolve_device(device)
    try:
        model = _load_model(str(checkpoint.resolve()))
    except Exception as error:
        LOGGER.exception("Unable to load YOLO checkpoint %s", checkpoint)
        raise RuntimeError(f"Unable to load YOLO checkpoint: {error}") from error
    try:
        results = model.predict(
            source=rgb_image[:, :, ::-1],
            conf=confidence_threshold,
            device=inference_device,
            verbose=False,
        )
    except Exception as error:
        LOGGER.exception("YOLO inference failed for %s", checkpoint)
        raise RuntimeError(f"YOLO inference failed: {error}") from error

    result = results[0]
    names = result.names
    class_names = tuple(str(value) for _, value in sorted(names.items())) if isinstance(names, dict) else tuple(str(value) for value in names)
    detections: list[ClothingDetection] = []
    height, width = rgb_image.shape[:2]
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            label = _class_name(names, class_id)
            if label is None:
                LOGGER.warning("Ignoring YOLO detection with invalid class id %s", class_id)
                continue
            coordinates = tuple(float(value) for value in boxes.xyxy[index].tolist())
            x1, y1, x2, y2 = coordinates
            normalized = tuple(round(value, 6) for value in (x1 / width, y1 / height, x2 / width, y2 / height))
            detections.append(ClothingDetection(
                label=label,
                class_id=class_id,
                confidence=round(float(boxes.conf[index].item()), 4),
                bbox=tuple(round(value, 2) for value in coordinates),
                normalized_bbox=normalized,
                category=_category_for(label),
            ))

    annotated_bgr = result.plot() if hasattr(result, "plot") else rgb_image
    annotated_image = annotated_bgr[:, :, ::-1] if hasattr(annotated_bgr, "shape") else annotated_bgr
    has_fashion_label = any(label.strip().lower() in _FASHION_LABELS for label in class_names)
    if has_fashion_label:
        status = "Clothing-trained YOLO checkpoint loaded."
    else:
        status = "Checkpoint loaded, but its classes are not validated as fashion categories."
    return ClothingDetectionResult(
        detections=detections,
        annotated_image=annotated_image,
        model_status=status,
        model_classes=class_names,
        device=inference_device,
        confidence_threshold=confidence_threshold,
    )


def detect_and_classify_clothing(
    image: Any,
    yolo_checkpoint: str | Path = "models/yolo11n.pt",
    fashionclip_model: str = "patrickjohncyh/fashion-clip",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str | None = None,
) -> ClothingDetectionResult:
    """Backward-compatible entry point for the Streamlit clothing scan."""
    del fashionclip_model
    return detect_clothing(image, yolo_checkpoint, confidence_threshold, device)
