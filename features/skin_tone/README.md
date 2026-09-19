# Skin tone analyzer

This feature detects the face with MediaPipe Face Landmarker, builds a landmark-defined facial skin mask, converts the selected pixels from BGR to LAB, groups usable pixels with K-Means, and reports the dominant lightness family with a confidence estimate.

The estimate is intentionally broad, not a medical or identity classification. Eye and mouth landmark polygons are excluded so their darker pixels do not distort the result.
