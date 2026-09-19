# AI Fashion Stylist

A Streamlit application for visual fashion analysis, wardrobe intelligence, and personalized outfit recommendations.

## Current status

The initial app shell is available in `app/main.py`. It includes profile uploads, wardrobe input, navigation, and a shared `UserProfile` contract. The ML feature modules are staged for incremental implementation.

## Run locally

Use Python 3.11, create a virtual environment, install `requirements.txt`, then run:

```text
streamlit run app/main.py
```

The first UI slice uses placeholder analysis states. Model checkpoints are intentionally not downloaded until the individual feature modules are implemented.
