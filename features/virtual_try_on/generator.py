from pathlib import Path
from typing import Optional


def generate_virtual_try_on(
    person_bytes: bytes,
    garment_bytes: bytes,
    output_path: str | Path = "data/generated/virtual_try_on.png",
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
) -> Path:
    """Generate a visual try-on preview from a person and garment reference.

    This uses SDXL image-to-image as a local, open model baseline. It is a
    visualization, not a measurement or a guaranteed pixel-accurate garment fit.
    A dedicated IDM-VTON checkpoint can replace this adapter later.
    """
    if not person_bytes or not garment_bytes:
        raise ValueError("Provide both a full-body photo and a clothing item photo.")

    try:
        import io
        import torch
        from diffusers import AutoPipelineForImage2Image
        from PIL import Image, ImageOps
    except ImportError as error:
        raise ImportError("Install diffusers, torch, and Pillow to generate try-on previews.") from error

    person = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    garment = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
    canvas_size = (1024, 1024)
    person_preview = ImageOps.contain(person, (700, 900))
    garment_preview = ImageOps.contain(garment, (260, 350))
    canvas = Image.new("RGB", canvas_size, (240, 238, 231))
    canvas.paste(person_preview, ((700 - person_preview.width) // 2, 35))
    canvas.paste(garment_preview, (735 + (230 - garment_preview.width) // 2, 340))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    prompt = (
        "fashion editorial virtual try-on, show the same person wearing the clothing item, "
        "preserve the person's face, body pose, hair, and camera framing, realistic fabric drape, "
        "accurate garment color and silhouette, clean studio lighting"
    )
    negative_prompt = "different person, extra limbs, distorted hands, cropped body, blurry garment, text, watermark"
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canvas,
        strength=0.52,
        guidance_scale=7.0,
        num_inference_steps=30,
    ).images[0]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination)
    return destination
