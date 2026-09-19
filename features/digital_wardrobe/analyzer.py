from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class WardrobeTagResult:
    category: str
    color: str
    pattern: str
    mask_path: Optional[str]
    embedding: Optional[list[float]]
    confidence: float
    model_status: str


def segment_and_tag_clothing(
    image_bytes: bytes,
    sam_checkpoint: str | Path | None = None,
    fashionclip_model: str = "patrickjohncyh/fashion-clip",
    output_dir: str | Path = "data/wardrobe/masks",
) -> WardrobeTagResult:
    """Segment an item with SAM 2 and tag it with FashionCLIP.

    Heavy model imports are intentionally lazy so the wardrobe editor still works
    while checkpoints are being installed.
    """
    if not image_bytes:
        raise ValueError("The clothing image is empty.")
    if not sam_checkpoint or not Path(sam_checkpoint).exists():
        raise FileNotFoundError("SAM 2 checkpoint is not configured.")

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as error:
        raise ImportError("Install SAM 2 to enable clothing segmentation.") from error

    try:
        import open_clip
    except ImportError as error:
        raise ImportError("Install open-clip-torch to enable FashionCLIP tagging.") from error

    # Model construction is kept behind this adapter because SAM 2 checkpoint
    # configuration varies by release and should be selected by the team.
    del SAM2ImagePredictor, open_clip, output_dir, fashionclip_model
    raise NotImplementedError("Configure the SAM 2 model config and FashionCLIP weights for inference.")
