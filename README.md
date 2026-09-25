# Smart Style

A CPU-only Streamlit application for personalized fashion analysis, outfit recommendations, wardrobe intelligence, and weekly planning.

## Main features

The application has exactly four main sections:

1. Style Profile: local portrait/body analysis, gender, explanations, and deterministic color guidance.
2. Outfit Recommendations: profile-aware outfit types and live retailer search links.
3. My Wardrobe: local image analysis, editable metadata, and persistent uploaded images.
4. Weekly Planner: user-selected wardrobe combinations, optional activities, repetition warnings, and saved plans.

No image generation or virtual try-on is used. The legacy generator is retained only as a disabled compatibility stub.

## Why This Is a CVIP Project

Recommendations are downstream of local image acquisition, preprocessing, feature extraction, and deterministic classification:

```text
				 USER IMAGE
				     |
				     v
			    IMAGE ACQUISITION
				     |
				     v
			     PREPROCESSING
				     |
		    +------------+------------+
		    v            v            v
		FACE ROI      SKIN ROI      BODY ROI
		    |            |            |
			 v            v            v
		 LANDMARK         LAB         POSE
	     FEATURES      FEATURES     FEATURES
		    |            |            |
		    v            v            v
	    FACE SHAPE    SKIN TONE    BODY SHAPE
		    +------------+------------+
				     v
			     USER PROFILE
				     |
				     v
		     PERSONALIZED RECOMMENDATIONS
				     |
				     v
			     WEEKLY PLANNER
```

Wardrobe images follow a second local pipeline:

```text
WARDROBE IMAGE -> RGB CONVERSION + RESIZE -> TWO-CLUSTER COLOR EXTRACTION
		   -> CATEGORY/STYLE CLASSIFICATION -> OCCASION MAPPING
		   -> PERSISTED WARDROBE FEATURES -> RECOMMENDATIONS
```

The face pipeline uses MediaPipe Face Landmarker geometry for normalized face, jaw,
cheek, and forehead ratios. The skin pipeline masks facial skin regions, excludes
eyes and lips, converts pixels to LAB, and uses deterministic clustering/statistics.
The body pipeline uses MediaPipe Pose shoulder, hip, torso, and leg measurements.
Wardrobe analysis is intentionally lightweight and CPU-safe: color features come
from image pixels, while category/style clues are conservative local filename
heuristics and remain manually editable. It does not claim deep-learning garment
recognition.

The Style Profile page exposes actual original/preprocessed images, face landmark
overlays, skin ROI overlays, pose overlays, and extracted feature values in
`View Image Analysis`. Lighting, framing, occlusion, and filename ambiguity can
reduce reliability; failed or uncertain analysis is surfaced rather than fabricated.

Navigation uses a professional fixed bottom bar with exactly four icon-labelled
sections: Style Profile, Outfit Recommendations, My Wardrobe, and Weekly Planner.

## Run locally

Use Python 3.11, create a CPU virtual environment, install `requirements.txt`, then run:

```text
streamlit run app/main.py
```
