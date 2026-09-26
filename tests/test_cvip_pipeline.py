import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from core.color_engine import recommend_colors
from core.schemas import UserProfile
from core.wardrobe_analysis import FASHION_PROMPTS, GENERAL_MODEL_ID, STYLE_PROMPTS, MARKET_PROMPTS, NotClothingImageError, analyze_wardrobe_image
from core.wardrobe_store import WardrobeStore
from features.body_shape.analyzer import _classify_shape
from features.face_shape.analyzer import _classify_shape as classify_face_shape
from features.style_guidance import suggest_body_shape_outfits


class CvipPipelineTests(unittest.TestCase):
    def test_each_undertone_has_at_least_eight_distinct_color_suggestions(self):
        for undertone in ("Warm", "Cool", "Neutral"):
            with self.subTest(undertone=undertone):
                colors = recommend_colors(undertone)
                self.assertGreaterEqual(len(colors), 8)
                self.assertEqual(len({color.name for color in colors}), len(colors))

    def test_outfit_suggestions_change_with_body_shape_and_occasion(self):
        pear = suggest_body_shape_outfits("Pear", "Formal")
        inverted = suggest_body_shape_outfits("Inverted triangle", "Formal")

        self.assertEqual(len(pear), 3)
        self.assertNotEqual([item.name for item in pear], [item.name for item in inverted])
        self.assertIn("formal", pear[0].reason.lower())
        self.assertIn("tailoring", pear[0].reason.lower())
        traditional = suggest_body_shape_outfits("Pear", "Traditional", "Female")
        self.assertNotEqual([item.name for item in pear], [item.name for item in traditional])
        self.assertIn("saree", traditional[1].name.lower())

    def test_wardrobe_analysis_uses_model_scores_not_filename(self):
        image = Image.fromarray(np.full((64, 96, 3), (20, 130, 70), dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        gate_details = {"model": GENERAL_MODEL_ID, "prediction": "Clothing", "decision": "accepted"}
        gate_scores = {label: 0.0 for label in FASHION_PROMPTS}
        gate_scores["Kurti"] = 0.82
        gate_scores["Other clothing item"] = 0.04
        style_scores = {label: 0.01 for label in STYLE_PROMPTS}
        style_scores["Ethnic"] = 0.76
        market_scores = {label: 0.01 for label in MARKET_PROMPTS}
        market_scores["Unknown"] = 0.92
        with (
            patch("core.wardrobe_analysis._general_gate", return_value=(True, gate_details)),
            patch("core.wardrobe_analysis._prompt_scores", side_effect=[gate_scores, style_scores, market_scores]),
            patch("core.wardrobe_analysis._clothing_region", return_value=(image, "test clothing crop")),
            patch("core.wardrobe_analysis._color_features", return_value=("Green", None, {"scope": "test crop"})),
        ):
            result = analyze_wardrobe_image(buffer.getvalue(), "farm_landscape.png")

        self.assertEqual(result["category"], "Top")
        self.assertEqual(result["subcategory"], "Kurti")
        self.assertEqual(result["color"], "Green")
        self.assertEqual(result["market_category"], "Unknown")
        self.assertIn("analysis_details", result["extracted_features"])

    def test_non_clothing_gate_rejects_before_fashion_model(self):
        image = Image.fromarray(np.full((32, 48, 3), (30, 120, 40), dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        gate_details = {"prediction": "Farm", "decision": "rejected; image was not saved"}
        with (
            patch("core.wardrobe_analysis._general_gate", return_value=(False, gate_details)),
            patch("core.wardrobe_analysis._prompt_scores") as fashion_model,
        ):
            with self.assertRaises(NotClothingImageError) as raised:
                analyze_wardrobe_image(buffer.getvalue(), "shirt.jpg")

        fashion_model.assert_not_called()
        self.assertEqual(raised.exception.analysis_details["final_result"], "No clothing item detected; not saved")

    def test_uncertain_fashion_scores_return_unable_to_determine(self):
        image = Image.fromarray(np.full((64, 96, 3), (100, 110, 120), dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        gate_details = {"model": GENERAL_MODEL_ID, "prediction": "Clothing", "decision": "accepted"}
        category_scores = {label: 0.04 for label in FASHION_PROMPTS}
        category_scores["Kurti"] = 0.16
        category_scores["Dress"] = 0.15
        style_scores = {label: 0.1 for label in STYLE_PROMPTS}
        market_scores = {label: 0.25 for label in MARKET_PROMPTS}
        with (
            patch("core.wardrobe_analysis._general_gate", return_value=(True, gate_details)),
            patch("core.wardrobe_analysis._prompt_scores", side_effect=[category_scores, style_scores, market_scores]),
            patch("core.wardrobe_analysis._clothing_region") as clothing_region,
        ):
            result = analyze_wardrobe_image(buffer.getvalue(), "unknown.png")

        self.assertEqual(result["subcategory"], "Unable to determine")
        self.assertEqual(result["category"], "Uncategorized")
        self.assertTrue(result["classification_uncertain"])
        clothing_region.assert_not_called()

    def test_profile_features_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            store = WardrobeStore(Path(directory) / "items.json")
            profile = UserProfile(
                gender="Female",
                face_shape="Oval",
                skin_tone="Medium",
                undertone="Warm",
                body_shape="Pear",
                preferred_colors=["Emerald green", "Navy blue"],
                analysis_date="2026-09-26",
                analysis_timestamp="2026-09-26T14:30:00+05:30",
                extracted_features={"face": {"face_length_width": 1.3}, "skin": {"lab_mean": [55.1, 12.2, 8.4]}, "body": {"shoulder_hip_ratio": 0.9}},
            )
            store.save_profile(profile)
            loaded = store.load_profile()
            self.assertEqual(loaded.gender, "Female")
            self.assertEqual(loaded.skin_tone, "Medium")
            self.assertEqual(loaded.undertone, "Warm")
            self.assertEqual(loaded.face_shape, "Oval")
            self.assertEqual(loaded.body_shape, "Pear")
            self.assertEqual(loaded.preferred_colors, ["Emerald green", "Navy blue"])
            self.assertEqual(loaded.analysis_date, "2026-09-26")
            self.assertEqual(loaded.analysis_timestamp, "2026-09-26T14:30:00+05:30")
            self.assertEqual(loaded.extracted_features["face"]["face_length_width"], 1.3)
            self.assertEqual(loaded.extracted_features["skin"]["lab_mean"], [55.1, 12.2, 8.4])
            self.assertEqual(loaded.extracted_features["body"]["shoulder_hip_ratio"], 0.9)
            self.assertTrue(store.profile_path.exists())

    def test_profile_loader_defaults_missing_and_invalid_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            store = WardrobeStore(Path(directory) / "items.json")
            store.profile_path.write_text(json.dumps({"gender": "Female", "preferred_colors": "not-a-list", "extracted_features": [], "future_field": True}), encoding="utf-8")

            loaded = store.load_profile()

            self.assertEqual(loaded.gender, "Female")
            self.assertIsNone(loaded.skin_tone)
            self.assertIsNone(loaded.analysis_timestamp)
            self.assertEqual(loaded.preferred_colors, [])
            self.assertEqual(loaded.extracted_features, {})
            self.assertEqual(loaded.occasion, "Everyday")

    def test_profile_loader_handles_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            store = WardrobeStore(Path(directory) / "items.json")
            store.profile_path.write_text("{not-json", encoding="utf-8")

            loaded = store.load_profile()

            self.assertEqual(loaded.analysis_count, 0)
            self.assertEqual(loaded.preferred_colors, [])
            self.assertEqual(loaded.extracted_features, {})

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

    def test_body_classifier_does_not_overclassify_moderate_shoulder_bias_as_inverted_triangle(self):
        shape, _, _, _ = _classify_shape(1.28, 1.0, 1.0, 1.1)
        self.assertNotEqual(shape, "Inverted triangle")


if __name__ == "__main__":
    unittest.main()