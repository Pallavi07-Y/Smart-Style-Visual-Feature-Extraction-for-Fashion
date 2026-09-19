from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class FaceShapeResult:
    shape: str
    confidence: float
    face_box: Tuple[float, float, float, float]


def analyze_face_shape(image_bytes: bytes, model_path: str | Path) -> FaceShapeResult:
    """Detect a face with MediaPipe Face Landmarker and estimate its shape."""
    checkpoint = Path(model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Face Landmarker model not found: {checkpoint}")

    import cv2
    import mediapipe as mp
    import numpy as np

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError("The uploaded file is not a readable image.")
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    base_options = mp.tasks.BaseOptions(model_asset_path=str(checkpoint))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face was detected. Use a clear, front-facing portrait.")

    landmarks = result.face_landmarks[0]
    points = np.array([(landmark.x, landmark.y) for landmark in landmarks], dtype=np.float32)
    shape, confidence = _classify_shape(points)
    return FaceShapeResult(
        shape=shape,
        confidence=confidence,
        face_box=_face_box(points),
    )


def _classify_shape(points):
    face_width = _distance(points, 234, 454)
    face_length = _distance(points, 10, 152)
    jaw_width = _distance(points, 172, 397)
    cheek_width = _distance(points, 123, 352)
    forehead_width = _distance(points, 103, 332)
    if min(face_width, face_length, jaw_width, cheek_width, forehead_width) == 0:
        raise ValueError("Face landmarks did not provide enough geometry for classification.")

    length_ratio = face_length / face_width
    jaw_ratio = jaw_width / cheek_width
    forehead_ratio = forehead_width / cheek_width

    if length_ratio > 1.55:
        shape = "Oblong"
    elif length_ratio < 1.18 and jaw_ratio > 0.84:
        shape = "Round"
    elif jaw_ratio > 0.9 and forehead_ratio > 0.88:
        shape = "Square"
    elif forehead_ratio > 1.04 and jaw_ratio < 0.82:
        shape = "Heart"
    elif cheek_width > forehead_width * 1.08 and cheek_width > jaw_width * 1.08:
        shape = "Diamond"
    else:
        shape = "Oval"

    confidence = min(0.96, 0.62 + abs(length_ratio - 1.35) * 0.2)
    return shape, round(float(confidence), 2)


def _distance(points, first: int, second: int) -> float:
    delta = points[first] - points[second]
    return float((delta[0] ** 2 + delta[1] ** 2) ** 0.5)


def _face_box(points) -> Tuple[float, float, float, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return tuple(round(float(value), 4) for value in (minimum[0], minimum[1], maximum[0], maximum[1]))
