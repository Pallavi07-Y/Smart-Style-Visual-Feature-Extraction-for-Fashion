from dataclasses import dataclass


@dataclass(frozen=True)
class OutfitSuggestion:
    name: str
    reason: str


_SHAPE_OUTFITS = {
    "pear": (
        ("Structured-shoulder top + straight trousers", "Shoulder detail balances the wider hip line; a straight leg keeps the lower line simple."),
        ("Cropped jacket + high-rise jeans", "A shorter layer defines the waist and draws attention upward."),
        ("A-line tunic + straight pants", "The A-line skims the hips while straight pants keep a clean vertical line."),
    ),
    "spoon": (
        ("Open-neck blouse + straight trousers", "An open neckline adds presence above the hip; straight trousers keep the outline fluid."),
        ("Cropped cardigan + A-line skirt", "A shorter layer gives gentle waist definition while the skirt drapes from the hip."),
        ("Structured kurta + relaxed straight pants", "A defined shoulder and relaxed leg add balance without a tight fit."),
    ),
    "inverted triangle": (
        ("Open-neck top + wide-leg trousers", "The open neckline softens the upper line; a wider leg adds balance below."),
        ("Unstructured overshirt + straight jeans", "A soft shoulder avoids extra bulk and the straight jean creates an even line."),
        ("V-neck knit + pleated trousers", "The neckline draws the eye vertically and pleats add movement through the lower half."),
    ),
    "athletic": (
        ("Textured top + tapered chinos", "Texture and a gentle taper add dimension while keeping the fit easy."),
        ("Wrap-front shirt + straight jeans", "The wrap suggests waist definition without rigid tailoring."),
        ("Layered overshirt + relaxed trousers", "Soft layers add shape and the relaxed leg keeps the silhouette comfortable."),
    ),
    "rectangle": (
        ("Wrap shirt + high-rise trousers", "A wrap and higher waist create gentle contour through the middle."),
        ("Cropped jacket + straight jeans", "The shorter hem marks the waist while the straight leg keeps the line clean."),
        ("Belted tunic + wide-leg pants", "A belt adds optional waist definition and the wider leg adds movement."),
    ),
    "balanced": (
        ("Belted shirt + straight trousers", "The clean lines follow balanced proportions; the belt adds optional definition."),
        ("Soft blazer + dark jeans", "Light structure keeps the outfit polished without adding shoulder bulk."),
        ("Column dress or kurta + optional belt", "A long vertical line works with balanced proportions; the belt is an optional accent."),
    ),
}

_SHAPE_LABELS = {
    "pear": "Hip-balancing",
    "spoon": "Hip-skimming",
    "inverted triangle": "Shoulder-balancing",
    "athletic": "Softly defined",
    "rectangle": "Waist-defining",
    "balanced": "Clean-line",
}

_OCCASION_FINISH = {
    "formal": "polished fabric and clean tailoring",
    "interview": "polished fabric and understated finishing",
    "presentation": "crisp fabric and structured finishing",
    "wedding": "festive fabric and occasion-appropriate detailing",
    "traditional": "a comfortable fabric and a considered drape",
    "travel": "breathable fabric and easy-moving layers",
    "outdoor": "breathable fabric and practical layers",
}

_OCCASION_LOOKS = {
    "formal": ("Tailored blazer + straight trousers", "Polished blouse + wide-leg trousers", "Clean-lined dress + cropped jacket"),
    "interview": ("Structured jacket + tailored trousers", "Button-front shirt + straight trousers", "Simple dress + polished layer"),
    "presentation": ("Crisp blazer + straight trousers", "Structured shirt + tailored wide-leg pants", "Clean-lined dress + light jacket"),
    "traditional": ("Kurta set + straight pants", "Saree + defined blouse", "Festive set + optional waistcoat"),
    "wedding": ("Festive kurta set", "Saree or lehenga + defined blouse", "Occasion dress + light structured layer"),
    "party": ("Dressy top + tailored trousers", "Draped dress", "Structured blouse + flowing skirt"),
    "college": ("Breathable top + straight jeans", "Kurta + relaxed straight pants", "Shirt + easy chinos"),
    "casual": ("Relaxed top + straight jeans", "Soft shirt + easy trousers", "Simple dress or kurta + flat shoes"),
    "travel": ("Breathable layers + relaxed trousers", "Soft top + straight pants", "Easy dress + light overshirt"),
    "outdoor": ("Light layers + practical trousers", "Breathable shirt + straight pants", "Relaxed top + comfortable jeans"),
}

_TRADITIONAL_LOOKS_BY_GENDER = {
    "female": ("Kurta set + straight pants", "Saree + defined blouse", "Festive lehenga + structured blouse"),
    "male": ("Straight-cut kurta + pajama", "Kurta + tailored trousers", "Kurta + structured waistcoat"),
}


def suggest_body_shape_outfits(body_shape: str | None, occasion: str = "Casual", gender: str | None = None) -> list[OutfitSuggestion]:
    """Suggest three adaptable outfit silhouettes from the locally estimated shape."""
    shape_key = (body_shape or "").strip().lower()
    aliases = {"inverted triangle": "inverted triangle", "athletic": "athletic", "rectangle": "rectangle", "pear": "pear", "spoon": "spoon"}
    shape_key = aliases.get(shape_key, "balanced")
    occasion_key = (occasion or "").strip().lower()
    finish = _OCCASION_FINISH.get(occasion_key, "comfortable fabric and simple layering")
    look_names = _OCCASION_LOOKS.get(occasion_key)
    if occasion_key in {"traditional", "wedding"}:
        look_names = _TRADITIONAL_LOOKS_BY_GENDER.get((gender or "").strip().lower(), look_names)
    suggestions = []
    for index, (name, reason) in enumerate(_SHAPE_OUTFITS[shape_key]):
        if look_names:
            name = f"{_SHAPE_LABELS[shape_key]}: {look_names[index]}"
        suggestions.append(OutfitSuggestion(name, f"{reason} For {occasion_key or 'casual'}, choose {finish}."))
    return suggestions