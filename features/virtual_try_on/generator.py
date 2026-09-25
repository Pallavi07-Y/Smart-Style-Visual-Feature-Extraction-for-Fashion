from pathlib import Path


def generate_virtual_try_on(
    person_bytes: bytes,
    garment_bytes: bytes,
    output_path: str | Path = "data/generated/virtual_try_on.png",
    model_id: str | None = None,
) -> Path:
    """Retained as a compatibility stub; generation is disabled by design."""
    del person_bytes, garment_bytes, output_path, model_id
    raise RuntimeError("Virtual try-on and image generation are disabled in Smart Style. Use actual wardrobe images instead.")
