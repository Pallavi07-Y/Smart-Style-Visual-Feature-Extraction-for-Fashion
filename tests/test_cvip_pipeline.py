import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from core.schemas import UserProfile
from core.wardrobe_analysis import analyze_wardrobe_image
from core.wardrobe_store import WardrobeStore
from features.body_shape.analyzer import _classify_shape
from features.face_shape.analyzer import _classify_shape as classify_face_shape


class CvipPipelineTests(unittest.TestCase):
    def test_wardrobe_analysis_extracts_color_and_category_features(self):
        image = Image.fromarray(np.full((32, 32, 3), (20, 130, 70), dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        result = analyze_wardrobe_image(buffer.getvalue(), "green_kurti.png")
        self.assertEqual(result["category"], "Kurti")
        self.assertEqual(result["color"], "Green")
        self.assertIn("color_proportions", result["extracted_features"])

    def test_profile_features_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            store = WardrobeStore(Path(directory) / "items.json")
            profile = UserProfile(gender="Female", face_shape="Oval", preferred_colors=["Emerald green"], extracted_features={"face": {"face_length_width": 1.3}})
            store.save_profile(profile)
            loaded = store.load_profile()
            self.assertEqual(loaded.gender, "Female")
            self.assertEqual(loaded.preferred_colors, ["Emerald green"])
            self.assertEqual(loaded.extracted_features["face"]["face_length_width"], 1.3)

    def test_face_classifier_returns_measurable_ratios(self):
        points = np.zeros((500, 2), dtype=np.float32)
        points[234], points[454] = (0.2, 0.5), (0.8, 0.5)
        points[10], points[152] = (0.5, 0.0), (0.5, 1.0)
        points[172], points[397] = (0.28, 0.65), (0.72, 0.65)
        points[123], points[352] = (0.25, 0.45), (0.75, 0.45)
        points[103], points[332] = (0.22, 0.3), (0.78, 0.3)
        shape, confidence, ratios = classify_face_shape(points)
        self.assertEqual(shape, "Oblong")
        self.assertGreater(confidence, 0)
        self.assertIn("face_length_width", ratios)

    def test_body_classifier_is_deterministic(self):
        shape, _, _, _ = _classify_shape(0.7, 1.0, 0.8, 1.2)
        self.assertEqual(shape, "Pear")


if __name__ == "__main__":
    unittest.main()