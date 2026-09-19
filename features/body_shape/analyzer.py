from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class BodyShapeResult:
    shape: str
    detail: str
    features: Tuple[str, ...]
    confidence: float
    shoulder_width: float
    hip_width: float
    shoulder_hip_ratio: float
    torso_length: float
    leg_length: float
    visibility: float
    landmark_box: Tuple[float, float, float, float]


def analyze_body_shape(image_bytes: bytes, model_path: str | Path) -> BodyShapeResult:
    """Estimate a broad body-shape family from MediaPipe Pose landmarks."""
    checkpoint = Path(model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Pose Landmarker model not found: {checkpoint}")

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
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raise ValueError("No full body was detected. Use a standing, full-body photo.")

    landmarks = result.pose_landmarks[0]
    shoulder_left, shoulder_right = landmarks[11], landmarks[12]
    hip_left, hip_right = landmarks[23], landmarks[24]
    ankle_left, ankle_right = landmarks[27], landmarks[28]
    knee_left, knee_right = landmarks[25], landmarks[26]
    visibility_values = [
        point.visibility
        for point in (shoulder_left, shoulder_right, hip_left, hip_right)
        if point.visibility is not None
    ]
    visibility = float(sum(visibility_values) / len(visibility_values))
    shoulder_width = _distance(shoulder_left.x, shoulder_left.y, shoulder_right.x, shoulder_right.y)
    hip_width = _distance(hip_left.x, hip_left.y, hip_right.x, hip_right.y)
    torso_length = _distance((shoulder_left.x + shoulder_right.x) / 2, (shoulder_left.y + shoulder_right.y) / 2, (hip_left.x + hip_right.x) / 2, (hip_left.y + hip_right.y) / 2)
    leg_points = (ankle_left, ankle_right) if _visible(ankle_left) and _visible(ankle_right) else (knee_left, knee_right)
    leg_length = (_distance(hip_left.x, hip_left.y, leg_points[0].x, leg_points[0].y) + _distance(hip_right.x, hip_right.y, leg_points[1].x, leg_points[1].y)) / 2
    if min(shoulder_width, hip_width, torso_length, leg_length) == 0 or visibility < 0.25:
        raise ValueError("Pose landmarks did not provide enough geometry for classification.")

    shape, detail, features, confidence = _classify_shape(shoulder_width, hip_width, torso_length, leg_length)
    points = np.array([(point.x, point.y) for point in landmarks], dtype=np.float32)
    return BodyShapeResult(
        shape=shape,
        detail=detail,
        features=features,
        confidence=round(min(confidence, visibility), 2),
        shoulder_width=round(shoulder_width, 4),
        hip_width=round(hip_width, 4),
        shoulder_hip_ratio=round(shoulder_width / hip_width, 3),
        torso_length=round(torso_length, 3),
        leg_length=round(leg_length, 3),
        visibility=round(visibility, 2),
        landmark_box=_landmark_box(points),
    )


def _classify_shape(shoulder_width: float, hip_width: float, torso_length: float, leg_length: float) -> Tuple[str, str, Tuple[str, ...], float]:
    ratio = shoulder_width / hip_width
    leg_torso_ratio = leg_length / torso_length
    if ratio >= 1.28:
        shape, detail, features = "Inverted triangle", "Your upper body carries more visual width than your lower body.", ("Shoulder line is the strongest feature.", "Structured jackets and open necklines can balance the silhouette.", "Bottoms with volume or detail can create visual balance.")
    elif ratio >= 1.12:
        shape, detail, features = "Athletic", "Your frame reads strong through the shoulders with a gently narrowing lower body.", ("Shoulders create a clean, athletic line.", "Straight and softly tailored garments are likely to sit well.", "Texture or detail at the hip can add balance.")
    elif ratio <= 0.78:
        shape, detail, features = "Pear", "Your lower body carries more visual width than your upper body.", ("Hip line is the strongest feature.", "Bright or detailed tops can draw the eye upward.", "Fluid or darker lower layers can create a long line.")
    elif ratio <= 0.9:
        shape, detail, features = "Spoon", "Your silhouette is softly fuller through the hips with a narrower shoulder line.", ("The hip line is gently rounded in proportion to the shoulders.", "Cropped jackets and defined necklines can add upper-body presence.", "Softly draped bottoms can follow the natural line.")
    elif 0.96 <= ratio <= 1.04:
        shape, detail, features = "Rectangle", "Your shoulders and hips create a fairly straight, balanced frame.", ("Shoulder and hip lines are visually aligned.", "Layering and gentle waist definition can add shape.", "Clean tailoring and column dressing can work especially well.")
    else:
        shape, detail, features = "Balanced", "Your shoulder and hip lines are close in visual width.", ("The frame reads proportionate from top to bottom.", "Both relaxed and tailored silhouettes can work.", "Use waist definition when you want more contour.")
    confidence = min(0.93, 0.62 + abs(ratio - 1.0) * 0.7 + min(0.12, abs(leg_torso_ratio - 1.5) * 0.08))
    return shape, detail, features, round(confidence, 2)


def _visible(point) -> bool:
    return point.visibility is None or point.visibility >= 0.25


def _distance(first_x: float, first_y: float, second_x: float, second_y: float) -> float:
    return ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5


def _landmark_box(points) -> Tuple[float, float, float, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return tuple(round(float(value), 4) for value in (minimum[0], minimum[1], maximum[0], maximum[1]))
