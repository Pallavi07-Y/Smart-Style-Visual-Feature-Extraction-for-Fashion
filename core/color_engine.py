from dataclasses import dataclass


@dataclass(frozen=True)
class ColorRecommendation:
    name: str
    score: int
    reason: str


_PALETTES = {
    "warm": (
        ("Emerald green", 94, "Rich green creates clear, warm contrast."),
        ("Terracotta", 92, "A warm earth tone echoes a golden undertone."),
        ("Cream", 88, "A soft neutral keeps the palette luminous."),
        ("Navy blue", 86, "Deep navy adds structure without harsh contrast."),
        ("Maroon", 84, "A deep red accent adds depth to warm coloring."),
    ),
    "cool": (
        ("Cobalt blue", 94, "A clear blue complements a cool undertone."),
        ("Berry", 92, "Berry tones add balanced, cool contrast."),
        ("Emerald green", 89, "Jewel green brings depth without warmth overload."),
        ("Plum", 87, "Plum is a grounded cool accent."),
        ("Soft white", 84, "A clean neutral keeps the palette crisp."),
    ),
    "neutral": (
        ("Emerald green", 93, "A jewel tone gives balanced contrast."),
        ("Navy blue", 91, "Navy is a versatile, low-risk anchor."),
        ("Maroon", 88, "Maroon adds depth while staying balanced."),
        ("Cream", 85, "Cream softens contrast without washing the palette out."),
        ("Cobalt blue", 83, "A clear accent adds energy to neutrals."),
    ),
}


def recommend_colors(undertone: str | None, skin_tone: str | None = None) -> list[ColorRecommendation]:
    """Return explainable colors from the detected undertone, never from UI literals."""
    del skin_tone
    key = (undertone or "neutral").strip().lower()
    palette = _PALETTES["warm" if "warm" in key else "cool" if "cool" in key else "neutral"]
    return [ColorRecommendation(name, score, reason) for name, score, reason in palette]


def colors_to_use_carefully(undertone: str | None) -> list[str]:
    key = (undertone or "neutral").strip().lower()
    if "warm" in key:
        return ["Icy grey", "Blue-based pastels", "Very cool lavender"]
    if "cool" in key:
        return ["Neon orange", "Yellow-beige", "Very warm camel"]
    return ["Very close-to-skin beige", "Highlighter neon", "Extreme head-to-toe contrast"]