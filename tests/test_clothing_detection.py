import unittest
from unittest.mock import patch

import numpy as np

from features.clothing_detection import detect_clothing, resolve_device
from features.clothing_detection import analyzer


class FakeValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBoxes:
    def __init__(self):
        self.cls = [FakeValue(0)]
        self.conf = [FakeValue(0.91)]
        self.xyxy = [FakeTensor([10, 20, 110, 80])]

    def __len__(self):
        return len(self.cls)

    def __getitem__(self, index):
        return self


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeResult:
    names = {0: "shirt"}

    def __init__(self, boxes=None):
        self.boxes = boxes

    def plot(self):
        return np.zeros((100, 200, 3), dtype=np.uint8)


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [self.result]


class ClothingDetectionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)
        self.checkpoint = "models/yolo11n.pt"

    def test_detection_schema_and_coordinates(self):
        model = FakeModel(FakeResult(FakeBoxes()))
        with patch.object(analyzer, "_load_model", return_value=model):
            result = detect_clothing(self.image, self.checkpoint, confidence_threshold=0.25, device="cpu")

        detection = result.detections[0]
        self.assertEqual(detection.label, "shirt")
        self.assertEqual(detection.category, "shirt")
        self.assertEqual(detection.class_id, 0)
        self.assertEqual(detection.bbox, (10.0, 20.0, 110.0, 80.0))
        self.assertEqual(detection.normalized_bbox, (0.05, 0.2, 0.55, 0.8))
        self.assertEqual(result.annotated_image.shape, self.image.shape)
        self.assertEqual(model.kwargs["conf"], 0.25)
        self.assertEqual(model.kwargs["device"], "cpu")

    def test_empty_result_is_clean(self):
        model = FakeModel(FakeResult(None))
        with patch.object(analyzer, "_load_model", return_value=model):
            result = detect_clothing(self.image, self.checkpoint)
        self.assertEqual(result.detections, [])

    def test_confidence_threshold_is_validated(self):
        with self.assertRaises(ValueError):
            detect_clothing(self.image, self.checkpoint, confidence_threshold=1.1)

    def test_invalid_image_is_rejected(self):
        with self.assertRaises((TypeError, ValueError)):
            detect_clothing(object(), self.checkpoint)

    def test_cpu_fallback(self):
        with patch.dict("sys.modules", {"torch": type("Torch", (), {"cuda": type("Cuda", (), {"is_available": staticmethod(lambda: False)})()})()}):
            self.assertEqual(resolve_device(), "cpu")


if __name__ == "__main__":
    unittest.main()