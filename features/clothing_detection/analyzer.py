from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ClothingDetection:
    category: str
    confidence: float
    box: tuple[float, float, float, float]
    embedding: Optional[list[float]] = None


@dataclass(frozen=True)
class ClothingDetectionResult:
    detections: list[ClothingDetection]
    model_status: str


def detect_and_classify_clothing(
    image_bytes: bytes,
    yolo_checkpoint: str | Path = "models/yolo11n.pt",
    fashionclip_model: str = "patrickjohncyh/fashion-clip",
) -> ClothingDetectionResult:
    """Detect clothing boxes with YOLO11 and classify/embed crops with FashionCLIP."""
    if not image_bytes:
        raise ValueError("The clothing image is empty.")
    checkpoint = Path(yolo_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"YOLO11 checkpoint not found: {checkpoint}")

    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("Install ultralytics and OpenCV to enable YOLO11 clothing detection.") from error

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The clothing image is not readable.")
    result = YOLO(str(checkpoint)).predict(source=image, verbose=False)[0]
    names = result.names
    detections = []
    for box in result.boxes:
        coordinates = box.xyxy[0].tolist()
        width, height = image.shape[1], image.shape[0]
        normalized_box = tuple(round(float(value), 4) for value in (coordinates[0] / width, coordinates[1] / height, coordinates[2] / width, coordinates[3] / height))
        class_id = int(box.cls[0])
        detections.append(ClothingDetection(category=str(names[class_id]), confidence=round(float(box.conf[0]), 3), box=normalized_box))
    status = "YOLO11 detections ready; FashionCLIP embeddings pending"
    if fashionclip_model:
        status += f" ({fashionclip_model})"
    return ClothingDetectionResult(detections=detections, model_status=status)
