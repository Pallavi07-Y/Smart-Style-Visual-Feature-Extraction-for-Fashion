from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class SkinToneResult:
    label: str
    shade_level: str
    undertone: str
    undertone_detail: str
    description: str
    color_guidance: Tuple[str, ...]
    lab_color: Tuple[float, float, float]
    chroma: float
    dominant_share: float
    confidence: float
    sample_count: int
    skin_pixel_mean_lab: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    skin_pixel_std_lab: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    face_landmarks: Tuple[Tuple[float, float], ...] = ()


def analyze_skin_tone(
    image_bytes: bytes,
    model_path: str | Path,
    clusters: int = 3,
) -> SkinToneResult:
    """Estimate skin tone from landmark-defined cheek and forehead regions."""
    checkpoint = Path(model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Face Landmarker model not found: {checkpoint}")

    import cv2
    import mediapipe as mp
    import numpy as np
    from sklearn.cluster import KMeans

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    height, width = image.shape[:2]
    face_mask = _skin_region_mask(landmarks, width, height, cv2, np)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab[face_mask > 0].astype(np.float32)
    pixels = pixels[(pixels[:, 0] > 35) & (pixels[:, 0] < 245)]
    if len(pixels) < clusters:
        raise ValueError("Not enough usable pixels were found in the face region.")

    model = KMeans(n_clusters=clusters, n_init=10, random_state=42)
    assignments = model.fit_predict(pixels)
    counts = np.bincount(assignments)
    dominant_index = int(np.argmax(counts))
    dominant_lab = model.cluster_centers_[dominant_index]
    share = float(counts[dominant_index] / len(assignments))
    shade_level = _tone_label(float(dominant_lab[0]))
    undertone, undertone_detail = _undertone(float(dominant_lab[1]), float(dominant_lab[2]))
    chroma = round(((float(dominant_lab[1]) - 128) ** 2 + (float(dominant_lab[2]) - 128) ** 2) ** 0.5, 2)
    label = f"{shade_level} / {undertone}"
    description, color_guidance = _tone_guidance(shade_level, undertone)

    return SkinToneResult(
        label=label,
        shade_level=shade_level,
        undertone=undertone,
        undertone_detail=undertone_detail,
        description=description,
        color_guidance=color_guidance,
        lab_color=tuple(round(float(value), 2) for value in dominant_lab),
        chroma=chroma,
        dominant_share=round(share, 2),
        confidence=round(min(0.99, 0.5 + share / 2), 2),
        sample_count=len(pixels),
        skin_pixel_mean_lab=tuple(round(float(value), 2) for value in pixels.mean(axis=0)),
        skin_pixel_std_lab=tuple(round(float(value), 2) for value in pixels.std(axis=0)),
        face_landmarks=tuple((round(float(landmark.x), 5), round(float(landmark.y), 5)) for landmark in landmarks),
    )


def _skin_region_mask(landmarks, width, height, cv2, np):
    points = np.array([(landmark.x * width, landmark.y * height) for landmark in landmarks], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    boundary = np.array([points[index] for index in _FACE_BOUNDARY], dtype=np.int32)
    cv2.fillPoly(mask, [boundary], 255)
    for exclusion in _EXCLUDED_FEATURES:
        cv2.fillPoly(mask, [np.array([points[index] for index in exclusion], dtype=np.int32)], 0)
    return mask


def skin_region_mask_from_normalized_landmarks(landmarks, width: int, height: int):
    """Build the same facial skin ROI used by analysis for visualization."""
    import cv2
    import numpy as np

    class Point:
        def __init__(self, x, y):
            self.x, self.y = x, y

    points = [Point(x, y) for x, y in landmarks]
    return _skin_region_mask(points, width, height, cv2, np)


# MediaPipe's stable face-mesh indices define the outer face and non-skin features.
_FACE_BOUNDARY = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67)
_EXCLUDED_FEATURES = (
    (33, 160, 158, 133, 153, 144),
    (263, 387, 385, 362, 380, 373),
    (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146),
)


def _tone_label(lightness: float) -> str:
    if lightness < 65:
        return "Very deep"
    if lightness < 90:
        return "Deep"
    if lightness < 115:
        return "Deep-medium"
    if lightness < 140:
        return "Medium-deep"
    if lightness < 165:
        return "Medium"
    if lightness < 190:
        return "Medium-light"
    if lightness < 215:
        return "Light"
    return "Very light"


def _undertone(a_value: float, b_value: float) -> Tuple[str, str]:
    a_delta = a_value - 128
    b_delta = b_value - 128
    if a_delta > 10 and b_delta > 12:
        return "Warm", "golden, peach, or red balance"
    if a_delta < 2 and b_delta < 8:
        return "Cool", "pink, rosy, or olive balance"
    if abs(a_delta) < 7 and abs(b_delta) < 12:
        return "Neutral", "low chroma balance"
    return "Neutral-warm", "slightly golden without a strong cast"


def _tone_guidance(shade_level: str, undertone: str) -> Tuple[str, Tuple[str, ...]]:
    shade_text = {
        "Very deep": "a very rich depth with strong color presence",
        "Deep": "a rich depth that carries saturated color well",
        "Deep-medium": "a deep-medium depth with clear color contrast",
        "Medium-deep": "a medium-deep depth with balanced contrast",
        "Medium": "a medium depth that adapts across many palettes",
        "Medium-light": "a medium-light depth with moderate contrast",
        "Light": "a light depth that responds well to both soft and vivid color",
        "Very light": "a very light depth where contrast can be especially noticeable",
    }.get(shade_level, "a distinctive natural depth")
    colors = {
        "Warm": ("terracotta", "olive", "camel", "cream", "coral"),
        "Cool": ("berry", "cobalt", "cool pink", "plum", "blue-grey"),
        "Neutral": ("true red", "emerald", "navy", "soft white", "charcoal"),
        "Neutral-warm": ("teal", "rust", "forest green", "ivory", "warm navy"),
    }.get(undertone, ("navy", "emerald", "soft white"))
    return f"Your complexion reads as {shade_text} with a {undertone.lower()} balance.", colors
