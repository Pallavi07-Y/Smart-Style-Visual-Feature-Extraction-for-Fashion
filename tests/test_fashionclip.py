import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from features.digital_wardrobe import (
    find_similar_items,
    get_fashion_embedding,
    get_fashion_text_similarity,
)
from features.digital_wardrobe import analyzer


class FakeProcessor:
    def __call__(self, text=None, images=None, return_tensors="pt", padding=False):
        values = {"pixel_values": torch.ones((1, 3, 2, 2))}
        if text is not None:
            values["input_ids"] = torch.ones((len(text), 2), dtype=torch.long)
            values["attention_mask"] = torch.ones((len(text), 2), dtype=torch.long)
        return values


class FakeModel:
    def get_image_features(self, **inputs):
        return torch.tensor([[3.0, 4.0]])

    def __call__(self, **inputs):
        class Output:
            logits_per_image = torch.tensor([[2.0, 1.0]])

        return Output()


class FashionClipTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_embedding_is_normalized(self):
        with patch.object(analyzer, "_load_fashionclip", return_value=(FakeProcessor(), FakeModel())):
            result = get_fashion_embedding(self.image, model_name="test-model", device="cpu")
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.device, "cpu")
        self.assertAlmostEqual(np.linalg.norm(result.embedding), 1.0, places=6)

    def test_text_similarity_uses_caller_labels(self):
        with patch.object(analyzer, "_load_fashionclip", return_value=(FakeProcessor(), FakeModel())):
            scores = get_fashion_text_similarity(self.image, ["shirt", "dress"], model_name="test-model", device="cpu")
        self.assertEqual(set(scores), {"shirt", "dress"})
        self.assertGreater(scores["shirt"], scores["dress"])

    def test_similarity_ranks_multiple_items(self):
        matches = find_similar_items(
            [1.0, 0.0],
            [
                {"item_id": "far", "embedding": [0.0, 1.0]},
                {"item_id": "near", "embedding": [1.0, 0.0]},
            ],
        )
        self.assertEqual([match.item_id for match in matches], ["near", "far"])
        self.assertEqual(matches[0].score, 1.0)

    def test_empty_wardrobe_is_clean(self):
        self.assertEqual(find_similar_items([1.0, 0.0], []), [])

    def test_invalid_image_is_rejected(self):
        with self.assertRaises((TypeError, ValueError)):
            get_fashion_embedding(object(), model_name="test-model", device="cpu")

    def test_embedding_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            find_similar_items([1.0, 0.0], [{"item_id": "bad", "embedding": [1.0]}])


if __name__ == "__main__":
    unittest.main()