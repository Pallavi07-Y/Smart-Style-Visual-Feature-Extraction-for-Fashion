import hashlib
from datetime import date
from pathlib import Path
import sys
from urllib.parse import quote_plus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from core.color_engine import colors_to_use_carefully, recommend_colors
from core.schemas import UserProfile, WardrobeItem
from core.scoring_engine import weighted_score
from core.wardrobe_analysis import analyze_wardrobe_image
from core.wardrobe_store import WardrobeStore
from features.body_shape import analyze_body_shape
from features.face_shape import analyze_face_shape
from features.planner import generate_weekly_plan
from features.recommendation import recommend_outfits
from features.skin_tone import analyze_skin_tone
from features.skin_tone.analyzer import skin_region_mask_from_normalized_landmarks

FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
POSE_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
WARDROBE_STORE = WardrobeStore(PROJECT_ROOT / "data" / "wardrobe" / "items.json")
IMAGE_DIR = PROJECT_ROOT / "data" / "wardrobe" / "images"
GENDERS = ["Female", "Male", "Prefer not to say"]
OCCASIONS = ["College", "Casual", "Formal", "Interview", "Presentation", "Party", "Wedding", "Traditional", "Dinner", "Travel", "Outdoor", "Other"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _index(options: list[str], value: str | None) -> int:
    return options.index(value) if value in options else 0


def _signature(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _profile() -> UserProfile:
    if "profile" not in st.session_state:
        st.session_state.profile = WARDROBE_STORE.load_profile()
    if "wardrobe_items" not in st.session_state:
        st.session_state.wardrobe_items = WARDROBE_STORE.load()
    profile = st.session_state.profile
    for field, default in (("gender", None), ("profile_image_path", None), ("analysis_date", None), ("extracted_features", {})):
        if not hasattr(profile, field):
            setattr(profile, field, default.copy() if isinstance(default, dict) else default)
    profile.wardrobe_items = st.session_state.wardrobe_items
    for item in profile.wardrobe_items:
        for field, default in (("secondary_color", None), ("date_added", None), ("suitable_occasions", []), ("extracted_features", {}), ("classification_uncertain", False)):
            if not hasattr(item, field):
                setattr(item, field, list(default) if isinstance(default, list) else default)
    _backfill_wardrobe_analysis(profile.wardrobe_items)
    return profile


def _backfill_wardrobe_analysis(items: list[WardrobeItem]) -> None:
    """Analyze legacy stored images once so old records are not left blank."""
    changed = False
    for item in items:
        image_path = _stored_image_path(item)
        needs_analysis = (
            image_path is not None
            and (item.classification_uncertain or "Awaiting" in (item.model_status or "") or not item.extracted_features)
        )
        if not needs_analysis:
            continue
        try:
            tags = analyze_wardrobe_image(image_path.read_bytes(), f"{item.name} {item.image_name or ''}")
        except (OSError, ImportError, TypeError, ValueError):
            continue
        for field, value in tags.items():
            setattr(item, field, value)
        if not item.date_added:
            item.date_added = date.today().isoformat()
        WARDROBE_STORE.update(item)
        changed = True
    if changed:
        st.session_state.wardrobe_items = WARDROBE_STORE.load()


def _stored_image_path(item: WardrobeItem) -> Path | None:
    candidates = []
    if item.image_path:
        candidates.append(Path(item.image_path))
    if item.image_name:
        candidates.extend((IMAGE_DIR / item.image_name, PROJECT_ROOT / "data" / "wardrobe" / item.image_name, PROJECT_ROOT / item.image_name))
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _image(item: WardrobeItem, caption: str | None = None) -> None:
    image_path = _stored_image_path(item)
    if image_path:
        st.image(str(image_path), caption=caption or item.name, use_column_width=True)
    else:
        st.caption(f"{caption or item.name}: original image is not stored")


def _score(value: float) -> str:
    return f"{round(value * 100):.0f}/100"


def _swatch_color(name: str) -> str:
    colors = {
        "emerald green": "#087f5b", "terracotta": "#c7654a", "cream": "#f3e5c1",
        "navy blue": "#183153", "maroon": "#7f1d35", "cobalt blue": "#2454c4",
        "berry": "#a32d63", "plum": "#6c3b72", "soft white": "#f7f7f2",
    }
    return colors.get(name.lower(), "#9aa39c")


def _image_quality_notes(image_bytes: bytes) -> list[str]:
    import cv2
    import numpy as np

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return ["The uploaded file is not a readable image."]
    notes = []
    height, width = image.shape[:2]
    if min(height, width) < 160:
        notes.append("The image is small; use a larger image for more reliable landmarks.")
    brightness = float(image.mean())
    if brightness < 35 or brightness > 225:
        notes.append("Lighting is extreme; skin-color estimates may be less reliable.")
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    if sharpness < 25:
        notes.append("The image appears soft or blurry; use a sharper capture if possible.")
    return notes


def _body_capture_notes(image_bytes: bytes) -> list[str]:
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return ["The body-camera frame is not readable."]
    height, width = image.shape[:2]
    notes = []
    if width > height:
        notes.append("Use the camera in portrait orientation so the full body length and shoulder-to-hip width fit in one frame.")
    if height < 720:
        notes.append("Move back and capture a taller, higher-resolution full-body frame for reliable pose landmarks.")
    return notes


def _save_profile_analysis(profile: UserProfile, portrait: bytes | None, body: bytes | None) -> None:
    if not portrait and not body:
        st.warning("Upload or capture at least one photo before analysis.")
        return
    for image_bytes in (portrait, body):
        if image_bytes:
            for note in _image_quality_notes(image_bytes):
                st.warning(note)
    if body:
        for note in _body_capture_notes(body):
            st.warning(note)
    face = tone = result = None
    if portrait:
        try:
            face = analyze_face_shape(portrait, FACE_MODEL)
            profile.face_shape = face.shape
            st.session_state.face_details = face
        except (FileNotFoundError, ImportError, ValueError) as error:
            st.warning(f"Face analysis could not be determined confidently: {error}")
        try:
            tone = analyze_skin_tone(portrait, FACE_MODEL)
            profile.skin_tone = tone.shade_level
            profile.undertone = tone.undertone
            st.session_state.tone_details = tone
        except (FileNotFoundError, ImportError, ValueError) as error:
            st.warning(f"Skin-tone analysis could not be determined confidently: {error}")
    if body:
        try:
            result = analyze_body_shape(body, POSE_MODEL)
            profile.body_shape = result.shape
            st.session_state.body_details = result
        except (FileNotFoundError, ImportError, ValueError) as error:
            st.warning(f"Body shape could not be determined confidently: {error}")
    if portrait:
        profile.profile_image_path = str(WARDROBE_STORE.save_image("profile", "profile.png", portrait, IMAGE_DIR))
    profile.analysis_date = date.today().isoformat()
    profile.preferred_colors = [color.name for color in recommend_colors(profile.undertone, profile.skin_tone)]
    profile.extracted_features = {
        "face": getattr(face, "feature_ratios", {}) if face else {},
        "skin": {"lab_mean": getattr(tone, "skin_pixel_mean_lab", ()), "lab_std": getattr(tone, "skin_pixel_std_lab", ()), "sample_count": getattr(tone, "sample_count", 0)} if tone else {},
        "body": {"shoulder_width": getattr(result, "shoulder_width", 0), "hip_width": getattr(result, "hip_width", 0), "shoulder_hip_ratio": getattr(result, "shoulder_hip_ratio", 0), "torso_length": getattr(result, "torso_length", 0), "leg_length": getattr(result, "leg_length", 0)} if result else {},
    }
    WARDROBE_STORE.save_profile(profile)
    st.session_state.profile = profile
    st.session_state.analysis_evidence = _build_analysis_evidence(portrait, body, face, tone, result)


def _decoded_rgb(image_bytes: bytes):
    import cv2

    encoded = cv2.imdecode(__import__("numpy").frombuffer(image_bytes, dtype=__import__("numpy").uint8), cv2.IMREAD_COLOR)
    if encoded is None:
        raise ValueError("The uploaded file is not a readable image.")
    return cv2.cvtColor(encoded, cv2.COLOR_BGR2RGB)


def _preprocessed_preview(image_bytes: bytes):
    import cv2

    image = _decoded_rgb(image_bytes)
    height, width = image.shape[:2]
    scale = min(1.0, 720 / max(height, width))
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
    return resized


def _face_overlay(image_bytes: bytes, face):
    import cv2

    image = _decoded_rgb(image_bytes).copy()
    height, width = image.shape[:2]
    for x, y in face.landmarks:
        cv2.circle(image, (int(x * width), int(y * height)), 1, (45, 109, 103), -1)
    x1, y1, x2, y2 = face.face_box
    cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (222, 104, 78), 2)
    return image


def _skin_overlay(image_bytes: bytes, tone):
    import cv2
    import numpy as np

    image = _decoded_rgb(image_bytes).copy()
    height, width = image.shape[:2]
    mask = skin_region_mask_from_normalized_landmarks(tone.face_landmarks, width, height)
    tint = np.zeros_like(image)
    tint[:, :, 1] = 190
    return np.where(mask[:, :, None] > 0, cv2.addWeighted(image, .55, tint, .45, 0), image)


def _body_overlay(image_bytes: bytes, body):
    import cv2

    image = _decoded_rgb(image_bytes).copy()
    height, width = image.shape[:2]
    for x, y in body.landmarks:
        cv2.circle(image, (int(x * width), int(y * height)), 3, (45, 109, 103), -1)
    x1, y1, x2, y2 = body.landmark_box
    cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (222, 104, 78), 2)
    return image


def _build_analysis_evidence(portrait, body, face, tone, body_result):
    evidence = []
    if portrait:
        evidence.extend([("Original portrait", portrait), ("Preprocessed portrait (RGB + resized)", _preprocessed_preview(portrait))])
        if face:
            evidence.append(("Face landmarks and detected face box", _face_overlay(portrait, face)))
        if tone and tone.face_landmarks:
            evidence.append(("Facial skin ROI used for LAB sampling", _skin_overlay(portrait, tone)))
    if body:
        evidence.append(("Original body image", body))
        evidence.append(("Preprocessed body image (RGB + resized)", _preprocessed_preview(body)))
        if body_result:
            evidence.append(("Pose landmarks and detected body box", _body_overlay(body, body_result)))
    return evidence


def _save_wardrobe_uploads(files) -> None:
    for upload in files or []:
        content = upload.getvalue()
        image_hash = _signature(content)
        if any(item.image_hash == image_hash for item in st.session_state.wardrobe_items):
            continue
        tags = analyze_wardrobe_image(content, upload.name)
        item = WardrobeItem(
            name=Path(upload.name).stem.replace("_", " ").title(),
            item_id="",
            image_name=upload.name,
            image_hash=image_hash,
            date_added=date.today().isoformat(),
            **tags,
        )
        item = WARDROBE_STORE.add(item)
        item.image_path = str(WARDROBE_STORE.save_image(item.item_id, upload.name, content, IMAGE_DIR))
        WARDROBE_STORE.update(item)
        st.session_state.wardrobe_items.append(item)


def _retailer_links(query: str) -> list[tuple[str, str]]:
    encoded = quote_plus(query)
    return [
        ("Amazon", f"https://www.amazon.in/s?k={encoded}"),
        ("Myntra", f"https://www.myntra.com/{encoded.replace('+', '-') }"),
        ("Flipkart", f"https://www.flipkart.com/search?q={encoded}"),
        ("AJIO", f"https://www.ajio.com/search/?text={encoded}"),
    ]


def _outfit_types(gender: str, occasion: str) -> list[tuple[str, str, str]]:
    female = {
        "College": [("Kurti + straight pants", "Semi-fitted", "Emerald Green"), ("Jeans + top", "Relaxed", "Navy Blue"), ("Oversized shirt + jeans", "Relaxed", "Cream")],
        "Formal": [("Formal trousers + shirt", "Tailored", "Navy Blue"), ("Blazer", "Structured", "Emerald Green"), ("Formal kurti", "Semi-fitted", "Maroon")],
        "Wedding": [("Saree", "Draped", "Emerald Green"), ("Anarkali", "Fluid", "Maroon"), ("Sharara", "Structured", "Navy Blue")],
        "Traditional": [("Saree", "Draped", "Emerald Green"), ("Anarkali", "Fluid", "Maroon"), ("Kurti set", "Semi-fitted", "Cream")],
    }
    male = {
        "College": [("Polo + chinos", "Relaxed", "Navy Blue"), ("Shirt + jeans", "Straight", "Emerald Green"), ("T-shirt + trousers", "Relaxed", "Cream")],
        "Formal": [("Formal shirt + trousers", "Tailored", "Navy Blue"), ("Blazer + trousers", "Structured", "Maroon"), ("Smart-casual shirt + chinos", "Straight", "Emerald Green")],
        "Wedding": [("Kurta", "Relaxed", "Emerald Green"), ("Nehru jacket + kurta", "Structured", "Maroon"), ("Formal ethnic combination", "Tailored", "Navy Blue")],
        "Traditional": [("Kurta", "Relaxed", "Emerald Green"), ("Nehru jacket + kurta", "Structured", "Maroon"), ("Formal ethnic combination", "Tailored", "Cream")],
    }
    common = [("Casual layered outfit", "Relaxed", "Navy Blue"), ("Polished separates", "Semi-fitted", "Emerald Green"), ("Comfortable tonal outfit", "Relaxed", "Cream")]
    return (female if gender == "Female" else male if gender == "Male" else {}).get(occasion, common)


st.set_page_config(page_title="Smart Style", page_icon="SS", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400,0,0..1');
:root { --ink:#1d2421; --muted:#68726d; --paper:#f5f4ef; --panel:#fffef9; --line:#d9ddd5; --coral:#de684e; --teal:#2e6d67; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; color:var(--ink); } .stApp { background:var(--paper); }
[data-testid="stSidebar"] { background:#e6ece5; border-right:1px solid var(--line); }
.brand { font-size:1.35rem; font-weight:800; letter-spacing:-.04em; } .brand-mark { color:var(--coral); } .eyebrow { color:var(--teal); font:500 .72rem 'DM Mono',monospace; letter-spacing:.11em; text-transform:uppercase; }
.hero { padding:1.1rem 0 1.8rem; border-bottom:1px solid var(--line); } .hero h1 { font-size:clamp(2.3rem,5vw,4.2rem); line-height:.98; letter-spacing:-.075em; max-width:760px; margin:.55rem 0 1rem; } .hero p { color:var(--muted); max-width:580px; line-height:1.6; }
.metric, .item-card { background:var(--panel); border:1px solid var(--line); padding:1rem; } .metric { min-height:100px; } .metric-label { color:var(--muted); font-size:.76rem; margin-top:.25rem; } .score { color:var(--coral); font-size:1.7rem; font-weight:800; }
.recommendation { background:var(--teal); color:#f8f8f1; padding:1.3rem; min-height:160px; } .recommendation p { color:#d9e8df; font-size:.85rem; line-height:1.55; }
.stButton > button { border-radius:0; border:1px solid var(--ink); background:var(--ink); color:white; font-weight:700; } .stButton > button:hover { background:var(--coral); border-color:var(--coral); color:white; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display:block; color:var(--ink); font-weight:800; margin-bottom:.8rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { display:flex; flex-direction:column; gap:.35rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label { display:flex; align-items:center; gap:.65rem; min-height:3.2rem; padding:.55rem .7rem; border:1px solid transparent; border-radius:6px; color:var(--ink); font-size:.82rem; font-weight:700; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover { background:#eef2ec; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) { background:#fffef9; border-color:var(--teal); color:var(--teal); }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label > div:first-child { display:none; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label::before { display:block; flex:0 0 1.5rem; font-family:'Material Symbols Outlined'; font-size:1.45rem; font-weight:400; line-height:1.2; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(1)::before { content:'person'; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(2)::before { content:'checkroom'; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(3)::before { content:'inventory_2'; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(4)::before { content:'calendar_month'; }
</style>
""", unsafe_allow_html=True)

profile = _profile()
navigation_options = ["Style Profile", "Outfit Recommendations", "My Wardrobe", "Weekly Planner"]
navigation_views = {option: option for option in navigation_options}
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">/</span> Smart Style</div>', unsafe_allow_html=True)
    st.caption("Computer vision for personal style.")
    st.divider()
    selected_navigation = st.radio("Navigation", navigation_options, key="sidebar_navigation")
    st.divider()
    st.markdown('<div class="eyebrow">PROFILE STATUS</div>', unsafe_allow_html=True)
    st.progress(profile.analysis_count / 3)
    st.caption(f"{profile.analysis_count}/3 visual signals ready")
view = navigation_views[selected_navigation]

st.markdown('<div class="hero"><div class="eyebrow">SMART STYLE / PERSONAL STUDIO</div><h1>Dress with a little more intention.</h1><p>Use your local style profile, real wardrobe images, and your own weekly rhythm to make outfit decisions.</p></div>', unsafe_allow_html=True)

if view == "Style Profile":
    st.markdown('<div class="eyebrow">FEATURE 01</div><h2>Style Profile</h2>', unsafe_allow_html=True)
    st.write("Use a clear front-facing image with good lighting. For better body-shape estimation, use an image where the upper/full body is visible.")
    profile.gender = st.selectbox("What is your gender?", GENDERS, index=_index(GENDERS, profile.gender), key="profile_gender")
    WARDROBE_STORE.save_profile(profile)
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        portrait_camera = st.camera_input("Use camera", key="portrait_camera")
        st.markdown("**Full-body camera capture**")
        st.caption("Hold the device vertically. Stand far enough away to include your head, shoulders, hips, feet, and both sides of your body with a little space around the frame.")
        body_camera = st.camera_input("Use camera for full-body image", key="body_camera")
        portrait_file = portrait_camera
        saved_portrait = Path(profile.profile_image_path) if profile.profile_image_path else None
        saved_portrait_bytes = saved_portrait.read_bytes() if saved_portrait and saved_portrait.exists() else None
        portrait_bytes = portrait_file.getvalue() if portrait_file else None
        body_bytes = body_camera.getvalue() if body_camera else None
        display_portrait = portrait_bytes or saved_portrait_bytes
        if display_portrait:
            st.image(display_portrait, caption="Actual camera profile image", width=300)
        if st.button("Analyze My Style", use_container_width=True):
            _save_profile_analysis(profile, portrait_bytes, body_bytes)
            st.success("Analysis completed where the local pipeline could determine a result.")
    with right:
        result_cards = [("Skin Tone", profile.skin_tone or "Not determined"), ("Undertone", profile.undertone or "Not determined"), ("Face Shape", profile.face_shape or "Not determined"), ("Body Shape", profile.body_shape or "Not determined")]
        explanations = {
            "Skin Tone": "Estimated from selected facial skin regions while reducing the influence of background and extreme lighting.",
            "Undertone": "Classified from the analyzed LAB skin-color characteristics used by the existing skin-tone analyzer.",
            "Face Shape": "Classified from facial landmark and proportion relationships detected by the existing face-analysis pipeline.",
            "Body Shape": "Classified from the detected shoulder, hip, torso, and leg landmark relationships in the existing pose pipeline.",
        }
        for label, value in result_cards:
            st.markdown(f'<div class="metric" style="margin-bottom:.55rem"><div class="metric-label">{label}</div><strong>{value}</strong></div>', unsafe_allow_html=True)
    with st.expander("View Image Analysis"):
        st.markdown("**Classification details**")
        for label, value in result_cards:
            st.markdown(f"**{label}:** {value}")
            st.caption(explanations[label])
        tone_details = st.session_state.get("tone_details")
        if tone_details:
            st.caption(tone_details.description)
        evidence = st.session_state.get("analysis_evidence", [])
        if evidence:
            st.caption("Only overlays generated from the current local landmark and skin-ROI results are shown.")
            for title, visual in evidence:
                st.markdown(f"**{title}**")
                st.image(visual, use_column_width=True)
        else:
            st.info("Run Analyze My Style to generate actual landmark and skin-region evidence for this session.")
        features = profile.extracted_features
        if features:
            st.markdown("**Extracted features used for classification**")
            st.json(features)
        st.caption("Pipeline: image acquisition -> RGB/BGR preprocessing -> facial/pose landmarks -> skin ROI and LAB statistics -> deterministic classification.")
    st.markdown('<div class="eyebrow">COLOR GUIDANCE</div><h3>Colors That Suit You</h3>', unsafe_allow_html=True)
    colors = recommend_colors(profile.undertone, profile.skin_tone)
    for row in range(0, len(colors), 3):
        columns = st.columns(3)
        for column, color in zip(columns, colors[row:row + 3]):
            with column:
                swatch = _swatch_color(color.name)
                st.markdown(f'<div class="metric"><span style="display:inline-block;width:1.2rem;height:1.2rem;border-radius:50%;background:{swatch};border:1px solid var(--line);vertical-align:middle;margin-right:.45rem"></span><strong>{color.name}</strong><div class="score">{color.score}/100</div><p>{color.reason}</p></div>', unsafe_allow_html=True)
                with st.expander("Why this color?"):
                    st.write("Recommended because it is included in the configured palette for the detected undertone and provides suitable contrast with the detected skin-tone category.")
                    st.caption("Good for: kurtis, dresses, sarees, tops, shirts, or accessories depending on the garment.")
    st.caption("Lower compatibility with your current color profile means a color may need more careful contrast or styling; it is not an appearance judgment.")
    st.write("Colors to use carefully: " + ", ".join(colors_to_use_carefully(profile.undertone)))

elif view == "Outfit Recommendations":
    st.markdown('<div class="eyebrow">FEATURE 02</div><h2>Outfit Recommendations</h2>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", GENDERS, index=_index(GENDERS, profile.gender), key="recommendation_gender")
    profile.gender = gender
    occasion = st.selectbox("What are you dressing for?", OCCASIONS, key="recommendation_occasion")
    profile.occasion = occasion
    st.caption("These are outfit types built from your profile, not universal ratings.")
    colors = recommend_colors(profile.undertone, profile.skin_tone)
    preferred_color = colors[0].name if colors else "Navy Blue"
    types = _outfit_types(gender, occasion)
    wardrobe_candidates = recommend_outfits(profile, st.session_state.wardrobe_items, occasion=occasion, top_k=3)
    st.markdown("### From Your Wardrobe")
    if wardrobe_candidates:
        for candidate in wardrobe_candidates:
            st.markdown(f"**{' + '.join(item.name for item in candidate.items)}**  ·  {_score(candidate.score)}")
            item_columns = st.columns(len(candidate.items))
            for column, item in zip(item_columns, candidate.items):
                with column:
                    _image(item)
                    st.caption(item.name)
            st.caption("This combination uses only analyzed items already stored in your wardrobe. " + " ".join(candidate.reasons))
    else:
        st.info("No complete analyzed wardrobe combination is available yet. Add wardrobe items for wardrobe-first recommendations.")
    st.markdown("### Outfit Types")
    for number, (outfit_type, fit, color) in enumerate(types[:3], 1):
        color = profile.preferred_colors[(number - 1) % len(profile.preferred_colors)] if profile.preferred_colors else color
        components = {"color_compatibility": .94 if color == preferred_color else .86, "body_shape_compatibility": .9 if profile.body_shape else .72, "occasion_compatibility": .92, "comfort_practicality": .86, "style_relevance": .84}
        suitability = weighted_score(components)
        left, right = st.columns([1.35, .65])
        with left:
            st.markdown(f'<div class="recommendation"><div class="eyebrow" style="color:#e8c56a">RECOMMENDATION {number}</div><h3>{color} {outfit_type}</h3><p><b>Type:</b> {outfit_type}<br><b>Color:</b> {color}<br><b>Fit:</b> {fit}<br><b>Occasion:</b> {occasion}</p><p>Why: {color} is in the saved preferred-color palette, the outfit responds to the saved {profile.body_shape or "available body-shape"} profile, and the type matches {occasion.lower()}. Face shape ({profile.face_shape or "not analyzed"}) is retained for future neckline/accessory refinement.</p></div>', unsafe_allow_html=True)
        with right:
            st.markdown(f'<div class="metric"><div class="metric-label">STYLE SUITABILITY SCORE</div><div class="score">{_score(suitability)}</div><p>Deterministic profile and occasion rules.</p></div>', unsafe_allow_html=True)
            query = f"{gender.lower()} {color} {outfit_type}"
            st.markdown("**Live retailer search**")
            for retailer, url in _retailer_links(query):
                st.markdown(f"[{retailer} search results]({url})")
            st.caption("Links open live retailer search results. No product name, image, price, or URL is invented by this app.")

elif view == "My Wardrobe":
    st.markdown('<div class="eyebrow">FEATURE 03</div><h2>My Wardrobe</h2>', unsafe_allow_html=True)
    uploads = st.file_uploader("Upload wardrobe images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="wardrobe_uploads")
    if st.button("Analyze and save wardrobe items", use_container_width=True):
        try:
            _save_wardrobe_uploads(uploads)
            st.success("Each uploaded image was analyzed locally and saved. Review or edit its tags below.")
        except (ImportError, TypeError, ValueError) as error:
            st.error(f"Wardrobe analysis failed: {error}")
    if not st.session_state.wardrobe_items:
        st.info("Upload actual clothing images to begin your wardrobe.")
    for item in list(st.session_state.wardrobe_items):
        with st.expander(item.name):
            image_column, data_column = st.columns([.35, .65])
            with image_column:
                _image(item, "Actual uploaded image")
                if _stored_image_path(item) is None:
                    restore = st.file_uploader("Restore original image", type=["jpg", "jpeg", "png"], key=f"restore_{item.item_id}")
                    if restore and st.button("Store image", key=f"store_image_{item.item_id}"):
                        content = restore.getvalue()
                        item.image_name = restore.name
                        item.image_hash = _signature(content)
                        item.image_path = str(WARDROBE_STORE.save_image(item.item_id, restore.name, content, IMAGE_DIR))
                        WARDROBE_STORE.update(item)
                        st.rerun()
                if item.classification_uncertain:
                    st.warning("Detection uncertain - please confirm the tags.")
                st.caption(item.model_status or "Local CV analysis")
            with data_column:
                item.category = st.text_input("Detected clothing type", item.category, key=f"category_{item.item_id}")
                item.color = st.text_input("Primary color", item.color or "", key=f"color_{item.item_id}") or None
                item.secondary_color = st.text_input("Secondary color", item.secondary_color or "", key=f"secondary_{item.item_id}") or None
                item.style = st.text_input("Style", item.style or "", key=f"style_{item.item_id}") or None
                item.suitable_occasions = st.multiselect("Suitable occasions", OCCASIONS + ["Family Gathering", "Festive Casual"], default=item.suitable_occasions, key=f"occasions_{item.item_id}")
                st.caption("Suitable occasions: " + (", ".join(item.suitable_occasions) or "Review manually"))
                st.markdown("**Extracted wardrobe features**")
                st.json(item.extracted_features or {"status": "No stored feature vector"})
                if st.button("Save edits", key=f"edit_{item.item_id}"):
                    WARDROBE_STORE.update(item)
                    st.success("Wardrobe metadata saved.")
                if st.button("Delete item", key=f"delete_{item.item_id}"):
                    WARDROBE_STORE.delete(item.item_id)
                    st.session_state.wardrobe_items = [stored for stored in st.session_state.wardrobe_items if stored.item_id != item.item_id]
                    st.rerun()

elif view == "Weekly Planner":
    st.markdown('<div class="eyebrow">FEATURE 04</div><h2>Weekly Planner</h2>', unsafe_allow_html=True)
    planning_mode = st.radio("Planning mode", ["AI Planning", "Manual Planning"], horizontal=True, key="planning_mode")
    if not st.session_state.wardrobe_items:
        st.info("Add wardrobe images first so the planner can use your actual clothes.")
    else:
        st.markdown("**Your wardrobe**")
        gallery = st.columns(min(5, len(st.session_state.wardrobe_items)))
        for column, item in zip(gallery, st.session_state.wardrobe_items):
            with column:
                _image(item)
                st.caption(f"{item.name}\n{item.category}")
        labels = {f"{item.name} [{item.item_id[:6]}]": item for item in st.session_state.wardrobe_items}
        saved_plan = WARDROBE_STORE.load_weekly_plan()
        selections = saved_plan.get("days", {}) if "planner_selections" not in st.session_state else st.session_state.planner_selections
        if planning_mode == "AI Planning":
            st.markdown("**AI planning suggestions**")
            ai_occasions, ai_activities = [], []
            for day in DAYS:
                day_left, day_middle = st.columns([.25, .75])
                with day_left:
                    st.markdown(f"**{day}**")
                with day_middle:
                    ai_occasions.append(st.selectbox("Occasion", OCCASIONS, key=f"ai_occasion_{day}"))
                    ai_activities.append(st.text_input("Activity / timetable", key=f"ai_activity_{day}", placeholder="College + lab, presentation, or family gathering"))
            if st.button("Generate AI wardrobe suggestions", use_container_width=True):
                st.session_state.ai_weekly_plan = generate_weekly_plan(profile, st.session_state.wardrobe_items, ai_occasions, DAYS, ai_activities)
            ai_plan = st.session_state.get("ai_weekly_plan")
            if ai_plan:
                for planned_day in ai_plan.days:
                    if not planned_day.outfit:
                        st.info(f"{planned_day.day}: no complete wardrobe combination was available.")
                        continue
                    st.markdown(f"**{planned_day.day}: {' + '.join(item.name for item in planned_day.outfit.items)}**")
                    ai_columns = st.columns(len(planned_day.outfit.items))
                    for column, item in zip(ai_columns, planned_day.outfit.items):
                        with column:
                            _image(item)
                    st.caption(f"Suggestion {_score(planned_day.outfit.score)} · {planned_day.reason} Select or change it below before saving.")
                    suggestion_labels = [label for label, item in labels.items() if item.item_id in {candidate.item_id for candidate in planned_day.outfit.items}]
                    keep_column, change_column = st.columns(2)
                    with keep_column:
                        if st.button(f"Keep {planned_day.day} suggestion", key=f"keep_{planned_day.day}"):
                            selections[planned_day.day] = suggestion_labels
                            st.session_state.planner_selections = selections
                            st.rerun()
                    with change_column:
                        if st.button(f"Change {planned_day.day} suggestion", key=f"change_{planned_day.day}"):
                            selections[planned_day.day] = []
                            st.session_state.planner_selections = selections
                            st.rerun()
        st.markdown("**Choose wardrobe items for each day. Selection is always yours.**")
        for day in DAYS:
            left, middle, right = st.columns([.22, .38, .4])
            with left:
                st.markdown(f"**{day}**")
            with middle:
                current_labels = [label for label in selections.get(day, []) if label in labels]
                selections[day] = st.multiselect("Outfit items", list(labels), default=current_labels, key=f"planner_items_{day}")
            with right:
                st.text_input("Activity / timetable (optional)", value=saved_plan.get("activities", {}).get(day, ""), key=f"activity_{day}")
        st.session_state.planner_selections = selections
        if st.button("Save Weekly Plan", use_container_width=True):
            plan = {"days": selections, "activities": {day: st.session_state.get(f"activity_{day}", "") for day in DAYS}, "saved_on": date.today().isoformat()}
            combinations = [tuple(sorted(values)) for values in selections.values() if values]
            repeated = {combo for combo in combinations if combinations.count(combo) > 1}
            WARDROBE_STORE.save_weekly_plan(plan)
            if repeated:
                st.warning("Plan saved. The same complete combination appears more than once; individual pieces may still be reused.")
            else:
                st.success("Weekly plan saved.")
        if st.button("Mark saved outfits as worn", use_container_width=True):
            selected_ids = {labels[label].item_id for values in selections.values() for label in values if label in labels}
            for item in st.session_state.wardrobe_items:
                if item.item_id in selected_ids:
                    item.last_worn = date.today().isoformat()
                    item.times_worn += 1
                    WARDROBE_STORE.update(item)
            st.success("Wear history updated for the saved wardrobe selections.")
        st.markdown('<div class="eyebrow">MY WEEKLY PLAN</div>', unsafe_allow_html=True)
        for day in DAYS:
            selected = [labels[label] for label in selections.get(day, []) if label in labels]
            if not selected:
                continue
            st.markdown(f"### {day}")
            st.caption(st.session_state.get(f"activity_{day}", saved_plan.get("activities", {}).get(day, "")) or "No activity added")
            item_columns = st.columns(len(selected))
            for column, item in zip(item_columns, selected):
                with column:
                    _image(item)
                    st.caption(item.name)