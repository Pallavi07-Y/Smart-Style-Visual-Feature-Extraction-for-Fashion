import hashlib
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote_plus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from core.color_engine import colors_to_use_carefully, recommend_colors
from core.schemas import UserProfile, WardrobeItem
from core.wardrobe_analysis import NotClothingImageError, analyze_wardrobe_image
from core.wardrobe_store import WardrobeStore
from features.body_shape import analyze_body_shape
from features.face_shape import analyze_face_shape
from features.planner import generate_weekly_plan
from features.recommendation import is_outfit_suitable_for_occasion, recommend_outfits
from features.skin_tone import analyze_skin_tone
from features.skin_tone.analyzer import skin_region_mask_from_normalized_landmarks
from features.style_guidance import suggest_body_shape_outfits

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
    for field, default in (("gender", None), ("profile_image_path", None), ("analysis_date", None), ("analysis_timestamp", None), ("extracted_features", {})):
        if not hasattr(profile, field):
            setattr(profile, field, default.copy() if isinstance(default, dict) else default)
    profile.wardrobe_items = st.session_state.wardrobe_items
    for item in profile.wardrobe_items:
        for field, default in (("secondary_color", None), ("date_added", None), ("suitable_occasions", []), ("extracted_features", {}), ("classification_uncertain", False), ("market_category", "Unknown")):
            if not hasattr(item, field):
                setattr(item, field, list(default) if isinstance(default, list) else default)
    return profile


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
        "olive green": "#78833d", "coral": "#e97968", "teal": "#287d78",
        "sapphire": "#2454a4", "rose": "#c96c83", "charcoal": "#41464b",
        "forest green": "#285b43", "dusty rose": "#bd858e",
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
    profile.analysis_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
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


def _save_wardrobe_uploads(files):
    analysis_results = []
    analysis_cache = st.session_state.setdefault("wardrobe_analysis_cache", {})
    pending_confirmations = st.session_state.setdefault("wardrobe_pending_confirmations", {})
    for upload in files or []:
        content = upload.getvalue()
        image_hash = _signature(content)
        if any(item.image_hash == image_hash for item in st.session_state.wardrobe_items):
            analysis_results.append({"filename": upload.name, "saved": False, "decision": "Duplicate image; not saved", "analysis_details": {}})
            continue
        if image_hash in analysis_cache:
            cached_result = analysis_cache[image_hash]
        else:
            try:
                cached_result = {"tags": analyze_wardrobe_image(content, upload.name)}
            except NotClothingImageError as error:
                cached_result = {"rejected": str(error), "analysis_details": error.analysis_details}
            except Exception as error:
                cached_result = {"error": str(error)}
            analysis_cache[image_hash] = cached_result
        if cached_result.get("rejected"):
            analysis_results.append({
                "filename": upload.name,
                "saved": False,
                "decision": "Rejected: no clothing item detected",
                "analysis_details": cached_result.get("analysis_details", {}),
            })
            continue
        if cached_result.get("error"):
            analysis_results.append({"filename": upload.name, "saved": False, "decision": "Analysis failed", "analysis_details": {"error": cached_result["error"]}})
            continue
        tags = cached_result["tags"]
        if tags.get("classification_uncertain"):
            pending_confirmations[image_hash] = {"filename": upload.name, "content": content, "tags": tags}
            analysis_results.append({
                "filename": upload.name,
                "saved": False,
                "pending_confirmation": True,
                "pending_id": image_hash,
                "decision": "Detection uncertain; confirm details before saving",
                "analysis_details": tags.get("extracted_features", {}).get("analysis_details", {}),
            })
            continue
        item = _persist_wardrobe_item(upload.name, content, image_hash, tags)
        analysis_results.append({
            "filename": upload.name,
            "saved": True,
            "decision": "Saved",
            "analysis_details": item.extracted_features.get("analysis_details", {}),
        })
    return analysis_results


def _persist_wardrobe_item(filename: str, content: bytes, image_hash: str, tags: dict, confirmation: dict | None = None) -> WardrobeItem:
    model_details = tags.get("extracted_features", {}).get("analysis_details", {})
    if confirmation:
        model_details = {**model_details, "user_confirmation": confirmation}
        tags = {**tags, "extracted_features": {**tags.get("extracted_features", {}), "analysis_details": model_details}}
    item = WardrobeItem(
        name=Path(filename).stem.replace("_", " ").title(),
        item_id="",
        image_name=filename,
        image_hash=image_hash,
        date_added=date.today().isoformat(),
        **tags,
    )
    item = WARDROBE_STORE.add(item)
    item.image_path = str(WARDROBE_STORE.save_image(item.item_id, filename, content, IMAGE_DIR))
    WARDROBE_STORE.update(item)
    st.session_state.wardrobe_items.append(item)
    return item


def _retailer_links(query: str) -> list[tuple[str, str, str]]:
    encoded = quote_plus(query)
    return [
        ("Amazon", f"https://www.amazon.in/s?k={encoded}", "amazon.in"),
        ("Myntra", f"https://www.myntra.com/search?q={encoded}", "myntra.com"),
        ("Flipkart", f"https://www.flipkart.com/search?q={encoded}", "flipkart.com"),
        ("AJIO", f"https://www.ajio.com/search/?text={encoded}", "ajio.com"),
        ("Tata CLiQ", f"https://www.tatacliq.com/search/?searchCategory=all&text={encoded}", "tatacliq.com"),
        ("Nykaa Fashion", f"https://www.nykaafashion.com/search/result/?q={encoded}", "nykaafashion.com"),
        ("Meesho", f"https://www.meesho.com/search?q={encoded}", "meesho.com"),
        ("Snapdeal", f"https://www.snapdeal.com/search?keyword={encoded}", "snapdeal.com"),
    ]


st.set_page_config(page_title="Smart Style", page_icon="SS", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400,0,0..1');
:root { --ink:#2a221d; --muted:#6b5a4f; --paper:#f7f2ee; --panel:#fffdfb; --line:#e5d8cc; --beige:#dcc1a5; --taupe:#b88a68; --soft-taupe:#efe3d7; --chestnut:#5b4233; --accent:#8b6b53; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; color:var(--ink); } .stApp { background:var(--paper); }
[data-testid="stSidebar"] { display:none; }
.main .block-container { padding-bottom:7rem; }
.brand { font-size:1.35rem; font-weight:800; letter-spacing:-.04em; } .brand-mark { color:var(--taupe); } .eyebrow { color:var(--accent); font:500 .72rem 'DM Mono',monospace; letter-spacing:.11em; text-transform:uppercase; }
.hero { padding:1.1rem 0 1.8rem; border-bottom:1px solid var(--line); } .hero h1 { font-size:clamp(2.3rem,5vw,4.2rem); line-height:.98; letter-spacing:-.075em; max-width:760px; margin:.55rem 0 1rem; } .hero p { color:var(--muted); max-width:580px; line-height:1.6; }
.metric, .item-card { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:1rem; box-shadow:0 8px 28px rgba(86,62,47,.05); } .metric { min-height:100px; } .metric-label { color:var(--muted); font-size:.76rem; margin-top:.25rem; } .score { color:var(--accent); font-size:1.7rem; font-weight:800; }
.recommendation { background:linear-gradient(135deg, var(--soft-taupe), var(--beige)); color:var(--ink); padding:1.3rem; min-height:160px; border:1px solid var(--line); border-radius:18px; box-shadow:0 10px 26px rgba(75,58,44,.08); } .recommendation p { color:var(--muted); font-size:.85rem; line-height:1.55; }
.retailer-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.35rem; margin:.4rem 0; }
.retailer-link { display:flex; align-items:center; gap:.45rem; min-height:2.4rem; padding:.4rem .5rem; border:1px solid var(--line); background:var(--panel); color:var(--ink)!important; text-decoration:none!important; font-size:.75rem; font-weight:700; border-radius:12px; }
.retailer-link:hover { border-color:var(--taupe); color:var(--accent)!important; }
.retailer-link img { width:20px; height:20px; flex:0 0 20px; object-fit:contain; }
.outfit-retailers { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.25rem; margin:.35rem 0 .6rem; }
.outfit-retailers .retailer-link { justify-content:center; min-height:1.9rem; padding:.25rem .3rem; font-size:.66rem; }
.analysis-link { display:inline-block; margin:.15rem 0 .7rem; color:var(--accent)!important; font-size:.82rem; font-weight:700; text-decoration:underline!important; text-underline-offset:3px; }
.color-option { display:flex; align-items:center; gap:.55rem; min-height:2.5rem; padding:.45rem .55rem; border:1px solid var(--line); background:var(--panel); font-size:.78rem; font-weight:700; border-radius:12px; }
.color-swatch { width:1.15rem; height:1.15rem; flex:0 0 1.15rem; border:1px solid #9aa39c; border-radius:50%; }
div[data-testid="stHorizontalBlock"]:has(.wardrobe-grid-marker) { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; align-items:start; }
div[data-testid="stHorizontalBlock"]:has(.wardrobe-grid-marker) > [data-testid="stColumn"] { width:auto!important; min-width:0; flex:none; padding:0; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-card-marker) { height:auto; min-height:0; border:1px solid rgba(184,138,104,.32); border-radius:14px; background:var(--panel); padding:.75rem; box-shadow:0 7px 22px rgba(86,62,47,.06); }
[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-card-marker) > div { gap:.45rem; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.wardrobe-card-marker) [data-testid="stImage"] img { width:100%; height:clamp(190px,24vw,260px); object-fit:contain; border-radius:9px; background:#f1eae3; }
.wardrobe-card-title { color:var(--ink); font-size:1rem; font-weight:800; line-height:1.3; }
.wardrobe-card-type { color:var(--muted); font-size:.78rem; font-weight:700; }
.wardrobe-card-meta { display:grid; grid-template-columns:1fr 1fr; gap:.55rem .75rem; padding-top:.35rem; border-top:1px solid var(--line); }
.wardrobe-meta-label { color:var(--muted); font-size:.65rem; font-weight:700; text-transform:uppercase; }
.wardrobe-meta-value { color:var(--ink); font-size:.75rem; line-height:1.4; overflow-wrap:anywhere; }
.wardrobe-image-fallback { display:flex; align-items:center; justify-content:center; min-height:190px; border:1px dashed var(--line); border-radius:10px; color:var(--muted); font-size:.8rem; background:#f7f2ee; }
.wardrobe-detail-title { color:var(--chestnut); font-size:1.25rem; font-weight:800; }
@media (max-width:1050px) { div[data-testid="stHorizontalBlock"]:has(.wardrobe-grid-marker) { grid-template-columns:repeat(3,minmax(0,1fr)); } }
@media (max-width:760px) { div[data-testid="stHorizontalBlock"]:has(.wardrobe-grid-marker) { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width:460px) { div[data-testid="stHorizontalBlock"]:has(.wardrobe-grid-marker) { grid-template-columns:1fr; } .wardrobe-card-meta { grid-template-columns:1fr; } }
.recommendation { padding:.9rem; min-height:0; }
div[data-testid="stHorizontalBlock"]:has(.bottom-nav-marker) { position:fixed; z-index:1000; left:0; bottom:0; width:100vw; box-sizing:border-box; padding:.55rem max(1rem, calc((100vw - 1280px) / 2)) .75rem; background:rgba(255,253,251,.96); backdrop-filter:blur(12px); border-top:1px solid var(--line); box-shadow:0 -8px 24px rgba(86,62,47,.06); }
div[data-testid="stHorizontalBlock"]:has(.bottom-nav-marker) [data-testid="stColumn"] { min-width:0; }
div[data-testid="stHorizontalBlock"]:has(.bottom-nav-marker) [data-testid="stBaseButton-secondary"],
div[data-testid="stHorizontalBlock"]:has(.bottom-nav-marker) [data-testid="stBaseButton-primary"] { width:100%; min-height:2.85rem; border-radius:12px; font-size:.8rem; font-weight:700; }
div[data-testid="stHorizontalBlock"]:has(.bottom-nav-marker) [data-testid="stBaseButton-primary"] { background:linear-gradient(135deg, var(--soft-taupe), var(--beige)); border-color:var(--taupe); color:var(--chestnut); box-shadow:0 5px 14px rgba(139,107,83,.12); }
@media (max-width: 680px) { .retailer-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
.stButton > button { border-radius:14px; border:1px solid var(--ink); background:var(--ink); color:white; font-weight:700; } .stButton > button:hover { background:var(--accent); border-color:var(--accent); color:white; }
div[data-testid="stHeaderActionElements"], div[data-testid="stStatusWidget"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

profile = _profile()
nav_labels = ["Profile", "Outfits", "Wardrobe", "Planner"]
nav_map = {"Profile": ("Style Profile", ":material/person:"), "Outfits": ("Outfit Recommendations", ":material/checkroom:"), "Wardrobe": ("My Wardrobe", ":material/inventory_2:"), "Planner": ("Weekly Planner", ":material/calendar_month:")}
if "selected_nav" not in st.session_state:
    st.session_state.selected_nav = "Profile"
analysis_expanded = st.session_state.pop("_open_analysis_requested", False)
view = nav_map[st.session_state.selected_nav][0]
if view in {"Outfit Recommendations", "My Wardrobe", "Weekly Planner"} and profile.analysis_count == 0:
    st.info("Complete Style Profile first.")

st.markdown('<div class="hero"><div class="eyebrow">SMART STYLE / PERSONAL STUDIO</div><h1>Dress with a little more intention.</h1><p>Use your local style profile, real wardrobe images, and your own weekly rhythm to make outfit decisions.</p></div>', unsafe_allow_html=True)
navigation_columns = st.columns(4, gap="small")
for index, (label, (page, icon)) in enumerate(nav_map.items()):
    with navigation_columns[index]:
        if index == 0:
            st.markdown('<span class="bottom-nav-marker"></span>', unsafe_allow_html=True)
        if st.button(label, key=f"main_nav_{label}", icon=icon, type="primary" if st.session_state.selected_nav == label else "secondary", use_container_width=True):
            st.session_state.selected_nav = label
            st.rerun()

if view == "Style Profile":
    st.markdown('<div class="eyebrow">FEATURE 01</div><h2>Style Profile</h2>', unsafe_allow_html=True)
    st.write("Use a clear front-facing image with good lighting. For better body-shape estimation, use an image where the upper/full body is visible.")
    if st.button("Open Image Analysis", key="open_profile_analysis"):
        st.session_state._open_analysis_requested = True
        st.rerun()
    profile.gender = st.selectbox("What is your gender?", GENDERS, index=_index(GENDERS, profile.gender), key="profile_gender")
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        portrait_camera = st.camera_input("Use camera", key="portrait_camera")
        st.markdown("**Full-body camera capture**")
        st.caption("Hold the device vertically. Stand far enough away to include your head, shoulders, hips, feet, and both sides of your body with a little space around the frame.")
        body_camera = st.camera_input("Use camera for full-body image", key="body_camera")
        portrait_file = portrait_camera
        portrait_bytes = portrait_file.getvalue() if portrait_file else None
        body_bytes = body_camera.getvalue() if body_camera else None
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
    colors = recommend_colors(profile.undertone, profile.skin_tone)
    profile_outfits = suggest_body_shape_outfits(profile.body_shape, "Casual", profile.gender)
    with st.expander("View Image Analysis", expanded=analysis_expanded):
        st.markdown("**Classification details**")
        for label, value in result_cards:
            st.markdown(f"**{label}:** {value}")
            st.caption(explanations[label])
        st.markdown("**Color guidance**")
        for color in colors:
            st.caption(f"{color.name}: {color.reason}")
        st.caption("Use carefully: " + "; ".join(colors_to_use_carefully(profile.undertone)) + ". These are styling suggestions, not rules.")
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
    st.caption("Eight colors selected from your undertone profile.")
    for row in range(0, len(colors), 4):
        columns = st.columns(4)
        for column, color in zip(columns, colors[row:row + 4]):
            with column:
                swatch = _swatch_color(color.name)
                st.markdown(f'<div class="color-option"><span class="color-swatch" style="background:{swatch}"></span>{escape(color.name)}</div>', unsafe_allow_html=True)
    st.caption("Use carefully: " + ", ".join(colors_to_use_carefully(profile.undertone)) + ".")
    st.markdown(f"### Outfit ideas for {profile.body_shape or 'a balanced starting shape'}")
    st.caption("Three easy-to-adapt silhouettes selected for your profile.")
    outfit_columns = st.columns(3)
    for number, (column, outfit) in enumerate(zip(outfit_columns, profile_outfits), 1):
        with column:
            st.markdown(f'<div class="recommendation"><div class="eyebrow" style="color:#e8c56a">IDEA {number}</div><h4>{escape(outfit.name)}</h4></div>', unsafe_allow_html=True)
            st.caption(outfit.reason)

elif view == "Outfit Recommendations":
    st.markdown('<div class="eyebrow">FEATURE 02</div><h2>Outfit Recommendations</h2>', unsafe_allow_html=True)
    profile_columns = st.columns(5, gap="small")
    profile_summary = (
        ("Gender", profile.gender or "Not set"),
        ("Skin tone", profile.skin_tone or "Not determined"),
        ("Undertone", profile.undertone or "Not determined"),
        ("Face shape", profile.face_shape or "Not determined"),
        ("Body shape", profile.body_shape or "Not determined"),
    )
    for column, (label, value) in zip(profile_columns, profile_summary):
        with column:
            st.markdown(f'<div class="metric"><div class="metric-label">{escape(label)}</div><strong>{escape(value)}</strong></div>', unsafe_allow_html=True)
    gender = st.selectbox("Recommendation gender", GENDERS, index=_index(GENDERS, profile.gender), key="recommendation_gender")
    occasion = st.selectbox("What are you dressing for?", OCCASIONS, index=_index(OCCASIONS, st.session_state.get("recommendation_occasion", profile.occasion)), key="recommendation_occasion")
    include_unisex_unknown = st.checkbox("Include Unisex and Unknown market-category items", value=True, key="recommendation_include_unisex_unknown")
    recommendation_profile = UserProfile(**{**profile.__dict__, "gender": gender, "occasion": occasion})
    if st.button("Generate Outfits", key="generate_wardrobe_outfits", icon=":material/auto_awesome:"):
        st.session_state.generated_wardrobe_outfits = recommend_outfits(
            recommendation_profile,
            st.session_state.wardrobe_items,
            occasion=occasion,
            top_k=8,
            complete_only=True,
            include_unisex_unknown=include_unisex_unknown,
        )
        st.session_state.generated_wardrobe_outfits_occasion = occasion
        st.session_state.generated_wardrobe_outfits_gender = gender
        st.session_state.find_similar_outfit_id = None

    generated = st.session_state.get("generated_wardrobe_outfits", [])
    generated_occasion = st.session_state.get("generated_wardrobe_outfits_occasion")
    generated_gender = st.session_state.get("generated_wardrobe_outfits_gender")
    if generated and (generated_occasion != occasion or generated_gender != gender):
        generated = []
    if not st.session_state.wardrobe_items:
        st.info("Upload actual wardrobe items before generating outfit combinations.")
    elif generated:
        if len(generated) < 5:
            st.info(f"Only {len(generated)} complete wardrobe combinations are currently available. Add more wardrobe items to generate additional outfits.")
        for number, candidate in enumerate(generated, 1):
            with st.container(border=True, key=f"recommendation_{candidate.outfit_id}"):
                st.markdown(f"### {number}. {escape(candidate.name)}")
                st.caption(f"{escape(candidate.occasion)} · Compatibility {_score(candidate.score)} · Actual wardrobe items")
                item_columns = st.columns(min(4, len(candidate.items)), gap="small")
                for index, item in enumerate(candidate.items):
                    with item_columns[index % len(item_columns)]:
                        _image(item)
                        st.caption(f"{item.subcategory or item.category} · {item.color or 'Color not determined'}")
                st.markdown("**Why this combination**")
                for reason in candidate.reasons:
                    st.caption(reason)
                action_columns = st.columns(2, gap="small")
                with action_columns[0]:
                    if st.button("Use My Wardrobe", key=f"use_outfit_{candidate.outfit_id}", icon=":material/calendar_add_on:"):
                        today = date.today()
                        week_start = today - timedelta(days=today.weekday())
                        day_options = [week_start + timedelta(days=offset) for offset in range(7)]
                        st.session_state.selected_nav = "Planner"
                        st.session_state.planner_week_start = week_start
                        st.session_state.planner_target_day = day_options[min(today.weekday(), 6)].isoformat()
                        st.session_state.planner_action = "add"
                        st.session_state.planner_detail_day = None
                        selections = st.session_state.setdefault("planner_temp_selection", {})
                        selections[st.session_state.planner_target_day] = list(candidate.item_ids)
                        st.rerun()
                with action_columns[1]:
                    if st.button("Find Similar", key=f"find_similar_{candidate.outfit_id}", icon=":material/search:"):
                        st.session_state.find_similar_outfit_id = candidate.outfit_id
                if st.session_state.get("find_similar_outfit_id") == candidate.outfit_id:
                    query = " ".join(
                        part for item in candidate.items
                        for part in (gender.lower() if gender != "Prefer not to say" else "", item.color or "", item.subcategory or item.category)
                    ) + f" {occasion.lower()}"
                    retailer_links = {name: url for name, url, _ in _retailer_links(query)}
                    similar_columns = st.columns(5, gap="small")
                    for column, retailer in zip(similar_columns, ("Amazon", "Flipkart", "Myntra", "Meesho", "AJIO")):
                        with column:
                            st.markdown(
                                f'<a class="retailer-link" href="{escape(retailer_links[retailer], quote=True)}" target="_blank" rel="noopener noreferrer">{retailer}</a>',
                                unsafe_allow_html=True,
                            )
                    st.caption("These links open retailer search results; products, prices, and availability are not verified by Smart Style.")
    elif st.session_state.get("generated_wardrobe_outfits") is not None:
        st.info(f"No complete wardrobe combinations match {occasion}. Add compatible tops/bottoms, dresses, shoes, or accessories and try again.")
    else:
        st.info("Choose an occasion and generate combinations from your saved wardrobe.")

    st.markdown(f"### Silhouette ideas for {occasion}")
    st.caption(f"Styling guidance based on the saved {profile.body_shape or 'body-shape profile'}; these ideas are not wardrobe items.")
    types = suggest_body_shape_outfits(profile.body_shape, occasion, gender)
    idea_columns = st.columns(3)
    for number, (column, outfit) in enumerate(zip(idea_columns, types), 1):
        with column:
            st.markdown(f'<div class="recommendation"><div class="eyebrow">IDEA {number}</div><h4>{escape(outfit.name)}</h4></div>', unsafe_allow_html=True)
            st.caption(outfit.reason)
    with st.expander("View Image Analysis: colors and outfit reasoning"):
        colors = recommend_colors(profile.undertone, profile.skin_tone)
        st.markdown(f"**Color profile:** {profile.undertone or 'Neutral / not analyzed'} undertone")
        for color in colors:
            st.caption(f"{color.name}: {color.reason}")
        st.markdown("**Colors to use carefully**")
        st.caption("; ".join(colors_to_use_carefully(profile.undertone)) + ". Try them as small accents if you enjoy them.")
        st.markdown(f"**Why these silhouettes for {profile.body_shape or 'a balanced starting shape'}?**")
        for outfit in types:
            st.markdown(f"**{outfit.name}**")
            st.caption(outfit.reason)
        st.caption("No Groq or other API key is required. Suggestions use local rules and saved profile data; the shop links open live web searches.")
    st.caption("Suggestions are deterministic styling guidance, not AI-generated fashion judgments.")

elif view == "My Wardrobe":
    st.markdown('<div class="eyebrow">FEATURE 03</div><h2>My Wardrobe</h2>', unsafe_allow_html=True)
    uploads = st.file_uploader("Upload wardrobe images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="wardrobe_uploads")
    if st.button("Analyze and save wardrobe items", use_container_width=True):
        try:
            st.session_state.wardrobe_analysis_results = _save_wardrobe_uploads(uploads)
            saved_count = sum(result["saved"] for result in st.session_state.wardrobe_analysis_results)
            rejected_count = sum(result["decision"].startswith("Rejected") for result in st.session_state.wardrobe_analysis_results)
            if saved_count:
                st.success(f"Saved {saved_count} clothing item(s) to your wardrobe.")
            if rejected_count:
                st.error(f"Rejected {rejected_count} image(s): no clothing item was detected; these images were not saved.")
            if not uploads:
                st.info("Choose one or more images to analyze.")
        except (ImportError, TypeError, ValueError) as error:
            st.error(f"Wardrobe analysis failed: {error}")
    for result in st.session_state.get("wardrobe_analysis_results", []):
        if result["decision"].startswith("Rejected"):
            st.error(f"{result['filename']}: {result['decision']}.")
        with st.expander(f"Analysis Details: {result['filename']}"):
            st.markdown(f"**Decision:** {escape(result['decision'])}")
            st.json(result["analysis_details"] or {"decision": result["decision"]})
    wardrobe_items = list(st.session_state.wardrobe_items)
    if not wardrobe_items:
        st.info("Upload actual clothing images to begin your wardrobe.")
    else:
        st.markdown("### Your Wardrobe")
        for row_start in range(0, len(wardrobe_items), 4):
            row_items = wardrobe_items[row_start:row_start + 4]
            columns = st.columns(4, gap="medium")
            with columns[0]:
                st.markdown('<span class="wardrobe-grid-marker"></span>', unsafe_allow_html=True)
            for item, column in zip(row_items, columns):
                uncertain = bool(item.classification_uncertain)
                clothing_type = "Unable to determine" if uncertain else (item.subcategory or item.category or "Unable to determine")
                category_value = "Unable to determine" if uncertain else (item.category or "Unable to determine")
                color_value = item.color or "Unable to determine"
                occasion_value = " • ".join(item.suitable_occasions) or "Unable to determine"
                style_value = item.style or "Unable to determine"
                if Path(item.name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    title_value = f"{item.color} {item.category}" if item.color and not uncertain else (item.category if not uncertain else "Wardrobe Item")
                else:
                    title_value = item.name
                with column:
                    with st.container(border=True, key=f"wardrobe_card_{item.item_id}"):
                        st.markdown('<span class="wardrobe-card-marker"></span>', unsafe_allow_html=True)
                        image_path = _stored_image_path(item)
                        if image_path:
                            st.image(str(image_path), use_column_width=True)
                        else:
                            st.markdown('<div class="wardrobe-image-fallback">Image unavailable</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="wardrobe-card-title">{escape(title_value)}</div>'
                            f'<div class="wardrobe-card-type">{escape(clothing_type)}</div>'
                            f'<div class="wardrobe-card-meta">'
                            f'<div><div class="wardrobe-meta-label">Color</div><div class="wardrobe-meta-value">{escape(color_value)}</div></div>'
                            f'<div><div class="wardrobe-meta-label">Secondary</div><div class="wardrobe-meta-value">{escape(item.secondary_color or "Not detected")}</div></div>'
                            f'<div><div class="wardrobe-meta-label">Occasions</div><div class="wardrobe-meta-value">{escape(occasion_value)}</div></div>'
                            f'<div><div class="wardrobe-meta-label">Style</div><div class="wardrobe-meta-value">{escape(style_value)}</div></div>'
                            f'<div><div class="wardrobe-meta-label">Category</div><div class="wardrobe-meta-value">{escape(category_value)}</div></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if uncertain:
                            st.caption("Detection uncertain. Review the details before relying on these tags.")
                        if st.button("View details", key=f"wardrobe_view_{item.item_id}", icon=":material/visibility:", use_container_width=True):
                            st.session_state.wardrobe_detail_item_id = item.item_id
                            st.rerun()

    detail_item_id = st.session_state.get("wardrobe_detail_item_id")
    detail_item = next((item for item in st.session_state.wardrobe_items if item.item_id == detail_item_id), None)
    if detail_item:
        detail_path = _stored_image_path(detail_item)
        detail_type = "Unable to determine" if detail_item.classification_uncertain else (detail_item.subcategory or detail_item.category or "Unable to determine")
        detail_title = detail_item.name
        if Path(detail_title).suffix.lower() in {".jpg", ".jpeg", ".png"}:
            detail_title = f"{detail_item.color} {detail_item.category}" if detail_item.color and not detail_item.classification_uncertain else (detail_item.category if not detail_item.classification_uncertain else "Wardrobe Item")
        try:
            date_added = date.fromisoformat(detail_item.date_added).strftime("%d %b %Y") if detail_item.date_added else "Not recorded"
        except ValueError:
            date_added = detail_item.date_added or "Not recorded"
        with st.container(border=True, key=f"wardrobe_detail_panel_{detail_item.item_id}"):
            st.markdown('<span class="wardrobe-detail-marker"></span>', unsafe_allow_html=True)
            st.markdown(f'<div class="wardrobe-detail-title">{escape(detail_title)}</div>', unsafe_allow_html=True)
            image_column, info_column = st.columns([1, 1.15], gap="large")
            with image_column:
                if detail_path:
                    st.image(str(detail_path), use_column_width=True)
                else:
                    st.markdown('<div class="wardrobe-image-fallback">Image unavailable</div>', unsafe_allow_html=True)
                    restore = st.file_uploader("Restore original image", type=["jpg", "jpeg", "png"], key=f"wardrobe_restore_{detail_item.item_id}")
                    if restore and st.button("Store image", key=f"wardrobe_store_image_{detail_item.item_id}"):
                        content = restore.getvalue()
                        detail_item.image_name = restore.name
                        detail_item.image_hash = _signature(content)
                        detail_item.image_path = str(WARDROBE_STORE.save_image(detail_item.item_id, restore.name, content, IMAGE_DIR))
                        WARDROBE_STORE.update(detail_item)
                        st.session_state.wardrobe_items = WARDROBE_STORE.load()
                        st.rerun()
            with info_column:
                st.markdown(f"**Clothing type:** {escape(detail_type)}")
                st.markdown(f"**Category:** {escape(detail_item.category if not detail_item.classification_uncertain else 'Unable to determine')}")
                st.markdown(f"**Market category:** {escape(detail_item.market_category or 'Unknown')}")
                st.markdown(f"**Color:** {escape(detail_item.color or 'Unable to determine')}")
                if detail_item.secondary_color:
                    st.markdown(f"**Secondary color:** {escape(detail_item.secondary_color)}")
                st.markdown(f"**Style:** {escape(detail_item.style or 'Unable to determine')}")
                st.markdown("**Suitable occasions**")
                st.markdown(" • ".join(escape(occasion) for occasion in detail_item.suitable_occasions) or "Unable to determine")
                st.markdown(f"**Date added:** {escape(date_added)}")
                st.markdown(f"**Last worn:** {escape(detail_item.last_worn or 'Not worn yet')}")
                st.markdown(f"**Times worn:** {detail_item.times_worn}")
                if detail_item.classification_uncertain:
                    st.warning("The existing analysis marked this item as uncertain. Confirm the type and tags before use.")
                detail_actions = st.columns(3, gap="small")
                with detail_actions[0]:
                    if st.button("Edit Details", key=f"wardrobe_edit_open_{detail_item.item_id}", icon=":material/edit:", use_container_width=True):
                        st.session_state.wardrobe_edit_item_id = detail_item.item_id
                        st.rerun()
                with detail_actions[1]:
                    if st.button("Remove from Wardrobe", key=f"wardrobe_remove_{detail_item.item_id}", icon=":material/delete:", use_container_width=True):
                        WARDROBE_STORE.delete(detail_item.item_id)
                        st.session_state.wardrobe_items = [item for item in st.session_state.wardrobe_items if item.item_id != detail_item.item_id]
                        st.session_state.wardrobe_detail_item_id = None
                        st.session_state.wardrobe_edit_item_id = None
                        st.rerun()
                with detail_actions[2]:
                    if st.button("Close", key=f"wardrobe_detail_close_{detail_item.item_id}", icon=":material/close:", use_container_width=True):
                        st.session_state.wardrobe_detail_item_id = None
                        st.session_state.wardrobe_edit_item_id = None
                        st.rerun()

            if st.session_state.get("wardrobe_edit_item_id") == detail_item.item_id:
                revisions = st.session_state.get("wardrobe_edit_revisions", {})
                revision = revisions.get(detail_item.item_id, 0)
                st.markdown("#### Edit details")
                name_value = st.text_input("Item name", value=detail_item.name, key=f"wardrobe_edit_name_{detail_item.item_id}_{revision}")
                category_value = st.text_input("Clothing type / category", value="" if detail_item.classification_uncertain else detail_item.category, key=f"wardrobe_edit_category_{detail_item.item_id}_{revision}")
                subcategory_value = st.text_input("Subcategory", value=detail_item.subcategory or "", key=f"wardrobe_edit_subcategory_{detail_item.item_id}_{revision}")
                market_category_value = st.selectbox(
                    "Garment market category",
                    ["Unknown", "Women's", "Men's", "Unisex"],
                    index=_index(["Unknown", "Women's", "Men's", "Unisex"], detail_item.market_category or "Unknown"),
                    key=f"wardrobe_edit_market_category_{detail_item.item_id}_{revision}",
                )
                edit_columns = st.columns(2, gap="medium")
                with edit_columns[0]:
                    color_value = st.text_input("Color", value=detail_item.color or "", key=f"wardrobe_edit_color_{detail_item.item_id}_{revision}")
                    style_value = st.text_input("Style", value=detail_item.style or "", key=f"wardrobe_edit_style_{detail_item.item_id}_{revision}")
                with edit_columns[1]:
                    secondary_value = st.text_input("Secondary color", value=detail_item.secondary_color or "", key=f"wardrobe_edit_secondary_{detail_item.item_id}_{revision}")
                    occasions_value = st.multiselect("Suitable occasions", OCCASIONS + ["Family Gathering", "Festive Casual"], default=detail_item.suitable_occasions, key=f"wardrobe_edit_occasions_{detail_item.item_id}_{revision}")
                if st.button("Save Details", key=f"wardrobe_edit_save_{detail_item.item_id}_{revision}", icon=":material/save:"):
                    detail_item.name = name_value.strip() or detail_item.name
                    detail_item.category = category_value.strip() or "Uncategorized"
                    detail_item.subcategory = subcategory_value.strip() or None
                    detail_item.market_category = market_category_value
                    detail_item.color = color_value.strip() or None
                    detail_item.secondary_color = secondary_value.strip() or None
                    detail_item.style = style_value.strip() or None
                    detail_item.suitable_occasions = occasions_value
                    detail_item.classification_uncertain = not bool(category_value.strip())
                    WARDROBE_STORE.update(detail_item)
                    st.session_state.wardrobe_items = WARDROBE_STORE.load()
                    revisions[detail_item.item_id] = revision + 1
                    st.session_state.wardrobe_edit_revisions = revisions
                    st.session_state.wardrobe_edit_item_id = None
                    st.rerun()

elif view == "Weekly Planner":
    st.markdown(
        """
        <style>
        .planner-shell { margin-top: .5rem; }
        .planner-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:1rem; }
        .planner-nav-btn { min-width:3rem; min-height:2.75rem; border:1px solid #e5d8cc; border-radius:14px; background:var(--panel); color:var(--ink); font-weight:800; }
        .planner-week-range { flex:1; text-align:center; color:var(--chestnut); font-size:1.1rem; font-weight:800; letter-spacing:-.02em; }
        .planner-grid-wrap { margin-top:.75rem; }
        div[data-testid="stHorizontalBlock"]:has(.planner-day-marker) { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; align-items:start; }
        div[data-testid="stHorizontalBlock"]:has(.planner-day-marker) > [data-testid="stColumn"] { width:auto!important; min-width:0; flex:none; padding:0; }
        .stVerticalBlock[class*="st-key-planner_day_"] > div > [data-testid="stVerticalBlockBorderWrapper"] { height:auto; min-height:0; border:1px solid rgba(184,138,104,.3); border-radius:16px; background:rgba(255,255,255,.48); padding:.8rem; }
        .stVerticalBlock[class*="st-key-planner_day_"] > div > [data-testid="stVerticalBlockBorderWrapper"] > div { gap:.45rem; }
        [data-testid="stColumn"]:has(.planner-day-marker) [data-testid="stImage"] img { width:100%; height:clamp(140px, 17vw, 200px); object-fit:contain; border-radius:10px; background:#f1eae3; }
        .planner-day-name { color:var(--muted); font-size:.74rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; }
        .planner-date { color:var(--chestnut); font-size:1.05rem; font-weight:800; }
        .planner-empty { display:flex; align-items:center; justify-content:center; flex-direction:column; width:100%; min-height:7rem; border:1px dashed rgba(184,138,104,.28); border-radius:12px; background:rgba(255,253,251,.42); color:var(--muted); }
        .planner-plus { font-size:1.8rem; line-height:1; font-weight:300; color:var(--taupe); }
        .planner-empty-title { margin-top:.2rem; font-weight:700; }
        .planner-item-title { color:var(--ink); font-size:.9rem; font-weight:800; }
        .planner-item-meta { color:var(--muted); font-size:.75rem; }
        [class*="st-key-planner_view_"] [data-testid="stBaseButton-secondary"] p,
        [class*="st-key-planner_change_"] [data-testid="stBaseButton-secondary"] p,
        [class*="st-key-planner_remove_"] [data-testid="stBaseButton-secondary"] p { display:none; }
        [class*="st-key-planner_view_"] [data-testid="stBaseButton-secondary"],
        [class*="st-key-planner_change_"] [data-testid="stBaseButton-secondary"],
        [class*="st-key-planner_remove_"] [data-testid="stBaseButton-secondary"] { min-height:2.4rem; padding:.35rem!important; }
        .planner-detail { margin-top:1.5rem; border:1px solid var(--line); background:var(--panel); border-radius:22px; padding:1rem; }
        .planner-detail-header { font-size:1.2rem; font-weight:800; color:var(--chestnut); }
        .planner-detail-grid { display:grid; grid-template-columns:1.1fr 1.3fr; gap:1rem; }
        .planner-detail-image { width:100%; border-radius:16px; overflow:hidden; border:1px solid var(--line); }
        .planner-detail-image img { width:100%; height:100%; object-fit:cover; max-height:420px; }
        .planner-modal { margin-top:1.2rem; border:1px solid var(--line); background:linear-gradient(180deg, var(--panel), rgba(255,255,255,.8)); border-radius:20px; padding:1rem; }
        .planner-modal-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:.75rem; }
        .planner-modal-title { color:var(--chestnut); font-size:1.1rem; font-weight:800; }
        .planner-gallery { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:.85rem; }
        .planner-gallery-card { border:1px solid var(--line); border-radius:16px; background:var(--panel); padding:.65rem; }
        .planner-gallery-card img { width:100%; aspect-ratio:4/5; object-fit:cover; border-radius:10px; }
        @media (max-width:900px) { div[data-testid="stHorizontalBlock"]:has(.planner-day-marker) { grid-template-columns:repeat(2,minmax(0,1fr)); } .planner-gallery { grid-template-columns:repeat(2,minmax(0,1fr)); } .planner-detail-grid { grid-template-columns:1fr; } }
        @media (max-width:520px) { div[data-testid="stHorizontalBlock"]:has(.planner-day-marker) { grid-template-columns:1fr; } .planner-gallery { grid-template-columns:1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    from datetime import timedelta

    if "planner_week_start" not in st.session_state:
        today = date.today()
        st.session_state.planner_week_start = today - timedelta(days=today.weekday())
    if "planner_target_day" not in st.session_state:
        st.session_state.planner_target_day = None
    if "planner_detail_day" not in st.session_state:
        st.session_state.planner_detail_day = None
    if "planner_temp_selection" not in st.session_state:
        st.session_state.planner_temp_selection = {}

    def _week_dates(start_date: date) -> list[date]:
        return [start_date + timedelta(days=offset) for offset in range(7)]

    def _planner_week_key(value: date) -> str:
        return value.isoformat()

    def _planner_load_week(week_start: date) -> tuple[dict, dict]:
        plan = WARDROBE_STORE.load_weekly_plan()
        weeks = plan.get("weeks", {}) if isinstance(plan, dict) else {}
        entry = weeks.get(_planner_week_key(week_start), {}) if isinstance(weeks, dict) else {}
        return entry.get("days", {}), entry.get("activities", {})

    def _planner_save_week(week_start: date, days: dict, activities: dict) -> None:
        plan = WARDROBE_STORE.load_weekly_plan()
        if not isinstance(plan, dict):
            plan = {}
        weeks = plan.setdefault("weeks", {})
        weeks[_planner_week_key(week_start)] = {"days": days, "activities": activities}
        WARDROBE_STORE.save_weekly_plan(plan)

    def _planner_day_item_ids(day_key: str, days: dict) -> list[str]:
        selected = days.get(day_key, [])
        if isinstance(selected, str):
            return [selected]
        if not isinstance(selected, list):
            return []
        return [item_id for item_id in selected if isinstance(item_id, str)]

    def _planner_day_items(day_key: str, days: dict) -> list[WardrobeItem]:
        item_ids = set(_planner_day_item_ids(day_key, days))
        return [item for item in st.session_state.wardrobe_items if item.item_id in item_ids]

    def _open_day_action(day_key: str, action: str) -> None:
        st.session_state.planner_target_day = day_key
        st.session_state.planner_detail_day = None
        st.session_state.planner_action = action
        if day_key not in st.session_state.planner_temp_selection:
            st.session_state.planner_temp_selection[day_key] = _planner_day_item_ids(day_key, _planner_load_week(st.session_state.planner_week_start)[0])

    week_start = st.session_state.planner_week_start
    week_dates = _week_dates(week_start)
    week_days, week_activities = _planner_load_week(week_start)
    if not isinstance(week_days, dict):
        week_days = {}
    if not isinstance(week_activities, dict):
        week_activities = {}

    header_left, header_center, header_right = st.columns([1, 3, 1])
    with header_left:
        if st.button("Previous Week", key="planner_prev_week", icon=":material/chevron_left:", use_container_width=True):
            st.session_state.planner_week_start = week_start - timedelta(days=7)
            st.session_state.planner_target_day = None
            st.session_state.planner_detail_day = None
            st.session_state.planner_temp_selection = {}
            st.rerun()
    with header_center:
        st.markdown(f"<div class='planner-week-range'>{week_dates[0].strftime('%d %b %Y')} – {week_dates[-1].strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
        if st.button("Current Week", key="planner_current_week", use_container_width=True):
            today = date.today()
            st.session_state.planner_week_start = today - timedelta(days=today.weekday())
            st.session_state.planner_target_day = None
            st.session_state.planner_detail_day = None
            st.session_state.planner_temp_selection = {}
            st.rerun()
    with header_right:
        if st.button("Next Week", key="planner_next_week", icon=":material/chevron_right:", use_container_width=True):
            st.session_state.planner_week_start = week_start + timedelta(days=7)
            st.session_state.planner_target_day = None
            st.session_state.planner_detail_day = None
            st.session_state.planner_temp_selection = {}
            st.rerun()

    if st.button("Save Week", use_container_width=True):
        _planner_save_week(week_start, week_days, week_activities)
        st.success("Saved")

    for row_start in (0, 4):
        row_dates = week_dates[row_start:row_start + 4]
        calendar_columns = st.columns(len(row_dates), gap="small")
        for current_day, day_column in zip(row_dates, calendar_columns):
            day_key = current_day.isoformat()
            chosen_items = _planner_day_items(day_key, week_days)
            with day_column:
                with st.container(border=True, key=f"planner_day_{day_key}"):
                    st.markdown('<span class="planner-day-marker"></span>', unsafe_allow_html=True)
                    st.markdown(f"<div class='planner-day-name'>{current_day.strftime('%A')}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='planner-date'>{current_day.strftime('%d %b')}</div>", unsafe_allow_html=True)
                    if chosen_items:
                        primary = chosen_items[0]
                        image_path = _stored_image_path(primary)
                        if image_path:
                            st.image(str(image_path), use_column_width=True)
                        else:
                            st.markdown("<div class='planner-empty'><div class='planner-plus'>+</div><div class='planner-empty-title'>No image</div></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='planner-item-title'>{escape(primary.name)}</div>", unsafe_allow_html=True)
                        if primary.category:
                            st.markdown(f"<div class='planner-item-meta'>{escape(primary.category)}</div>", unsafe_allow_html=True)
                        if len(chosen_items) > 1:
                            st.caption(f"{len(chosen_items)} pieces selected")
                        if week_activities.get(day_key):
                            st.caption(escape(week_activities[day_key]))
                        action_columns = st.columns(3, gap="small")
                        with action_columns[0]:
                            if st.button("View", key=f"planner_view_{day_key}", icon=":material/visibility:", help="View outfit details", use_container_width=True):
                                st.session_state.planner_detail_day = day_key
                                st.rerun()
                        with action_columns[1]:
                            if st.button("Change", key=f"planner_change_{day_key}", icon=":material/edit:", help="Change outfit", use_container_width=True):
                                _open_day_action(day_key, "change")
                                st.rerun()
                        with action_columns[2]:
                            if st.button("Remove", key=f"planner_remove_{day_key}", icon=":material/delete:", help="Remove outfit", use_container_width=True):
                                week_days.pop(day_key, None)
                                week_activities.pop(day_key, None)
                                _planner_save_week(week_start, week_days, week_activities)
                                st.rerun()
                    else:
                        st.markdown("<div class='planner-empty'><div class='planner-plus'>+</div><div class='planner-empty-title'>Add Outfit</div></div>", unsafe_allow_html=True)
                        if st.button("Add Outfit", key=f"planner_add_{day_key}", use_container_width=True):
                            _open_day_action(day_key, "add")
                            st.rerun()

    detail_day = st.session_state.get("planner_detail_day")
    if detail_day:
        detail_items = _planner_day_items(detail_day, week_days)
        if detail_items:
            st.markdown("<div class='planner-detail'>", unsafe_allow_html=True)
            st.markdown(f"<div class='planner-detail-header'>{date.fromisoformat(detail_day).strftime('%A')} — {date.fromisoformat(detail_day).strftime('%d %b')}</div>", unsafe_allow_html=True)
            detail_cols = st.columns([1.2, 1.5])
            with detail_cols[0]:
                primary = detail_items[0]
                image_path = _stored_image_path(primary)
                if image_path:
                    st.markdown("<div class='planner-detail-image'>", unsafe_allow_html=True)
                    st.image(str(image_path), use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            with detail_cols[1]:
                for item in detail_items:
                    st.markdown(f"**{escape(item.name)}**")
                    if item.category:
                        st.caption(f"Category: {item.category}")
                    if item.color:
                        st.caption(f"Color: {item.color}")
                    if item.style:
                        st.caption(f"Style: {item.style}")
                    if item.suitable_occasions:
                        st.caption(f"Suitable occasions: {', '.join(item.suitable_occasions)}")
                    if item.last_worn:
                        st.caption(f"Last worn: {item.last_worn}")
                    if item.times_worn:
                        st.caption(f"Times worn: {item.times_worn}")
                if st.button("Change Outfit", key=f"planner_detail_change_{detail_day}", use_container_width=True):
                    _open_day_action(detail_day, "change")
                    st.session_state.planner_detail_day = None
                    st.rerun()
                if st.button("Remove Outfit", key=f"planner_detail_remove_{detail_day}", use_container_width=True):
                    week_days.pop(detail_day, None)
                    week_activities.pop(detail_day, None)
                    _planner_save_week(week_start, week_days, week_activities)
                    st.session_state.planner_detail_day = None
                    st.rerun()
                if st.button("Close", key=f"planner_detail_close_{detail_day}", use_container_width=True):
                    st.session_state.planner_detail_day = None
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    target_day = st.session_state.get("planner_target_day")
    if target_day:
        st.markdown("<div class='planner-modal'>", unsafe_allow_html=True)
        st.markdown(f"<div class='planner-modal-header'><div class='planner-modal-title'>Add Outfit — {escape(date.fromisoformat(target_day).strftime('%A, %d %b'))}</div></div>", unsafe_allow_html=True)
        option = st.radio("Choose an option", ["Add from My Wardrobe", "Add recommended outfit", "Upload from Device"], horizontal=True, key=f"planner_option_{target_day}")
        activity_value = week_activities.get(target_day, "")
        activity_input = st.text_input("Activity / timetable (optional)", value=activity_value, key=f"planner_activity_{target_day}")
        week_activities[target_day] = activity_input
        if option == "Add from My Wardrobe":
            if not st.session_state.wardrobe_items:
                st.info("Upload wardrobe images first so you can plan with your actual clothes.")
            else:
                selected_ids = list(st.session_state.planner_temp_selection.get(target_day, []))
                for item in st.session_state.wardrobe_items:
                    selected = item.item_id in selected_ids
                    image_path = _stored_image_path(item)
                    st.markdown("<div class='planner-gallery-card'>", unsafe_allow_html=True)
                    if image_path:
                        st.image(str(image_path), use_column_width=True)
                    st.markdown(f"<div style='margin-top:.5rem; font-weight:700; color:var(--ink);'>{escape(item.name)}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:.72rem; color:var(--muted);'>{escape(item.category)}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:.72rem; color:var(--muted);'>{escape(item.color or 'Color not detected')}</div>", unsafe_allow_html=True)
                    if st.button("Selected" if selected else "Select item", key=f"planner_select_{item.item_id}_{target_day}", use_container_width=True):
                        current = list(st.session_state.planner_temp_selection.get(target_day, []))
                        if item.item_id in current:
                            current.remove(item.item_id)
                        else:
                            current.append(item.item_id)
                        st.session_state.planner_temp_selection[target_day] = current
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                if st.button("Add to selected day", key=f"planner_add_to_day_{target_day}", use_container_width=True):
                    selected_ids = st.session_state.planner_temp_selection.get(target_day, [])
                    if selected_ids:
                        week_days[target_day] = selected_ids
                        _planner_save_week(week_start, week_days, week_activities)
                        st.session_state.planner_target_day = None
                        st.session_state.planner_temp_selection.pop(target_day, None)
                        st.rerun()
                    else:
                        st.warning("Select at least one wardrobe item.")
        elif option == "Add recommended outfit":
            recommendation_occasion = st.selectbox("Outfit occasion", OCCASIONS, key=f"planner_recommendation_occasion_{target_day}")
            planner_recommendations = recommend_outfits(
                profile,
                st.session_state.wardrobe_items,
                occasion=recommendation_occasion,
                top_k=8,
                complete_only=True,
                include_unisex_unknown=True,
            )
            if not planner_recommendations:
                st.info(f"No complete wardrobe combinations are available for {recommendation_occasion.lower()}.")
            else:
                selected_recommendation = st.radio(
                    "Choose a wardrobe combination",
                    range(len(planner_recommendations)),
                    format_func=lambda index: planner_recommendations[index].name,
                    key=f"planner_recommendation_choice_{target_day}",
                )
                candidate = planner_recommendations[selected_recommendation]
                st.caption(" ".join(candidate.reasons))
                recommendation_columns = st.columns(min(4, len(candidate.items)), gap="small")
                for index, item in enumerate(candidate.items):
                    with recommendation_columns[index % len(recommendation_columns)]:
                        _image(item)
                        st.caption(item.subcategory or item.category)
                if st.button("Add recommended outfit to this day", key=f"planner_recommendation_add_{target_day}", icon=":material/calendar_add_on:"):
                    week_days[target_day] = list(candidate.item_ids)
                    _planner_save_week(week_start, week_days, week_activities)
                    st.session_state.planner_target_day = None
                    st.session_state.planner_temp_selection.pop(target_day, None)
                    st.rerun()
        else:
            uploaded = st.file_uploader("Upload outfit image", type=["jpg", "jpeg", "png"], key=f"planner_upload_{target_day}")
            if uploaded:
                content = uploaded.getvalue()
                st.image(content, use_column_width=True)
                upload_hash = _signature(content)
                cached_uploads = st.session_state.setdefault("planner_upload_analysis", {})
                if upload_hash not in cached_uploads:
                    try:
                        cached_uploads[upload_hash] = {"tags": analyze_wardrobe_image(content, uploaded.name)}
                    except NotClothingImageError as error:
                        cached_uploads[upload_hash] = {"rejected": str(error), "analysis_details": error.analysis_details}
                    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                        cached_uploads[upload_hash] = {"error": str(error)}
                upload_analysis = cached_uploads[upload_hash]
                if upload_analysis.get("rejected"):
                    st.error("Cannot add this image. No clothing was detected, so it was not saved.")
                    with st.expander("Analysis Details: rejected planner upload"):
                        st.json(upload_analysis.get("analysis_details", {}))
                elif upload_analysis.get("error"):
                    st.error(f"Wardrobe analysis failed: {upload_analysis['error']}")
                else:
                    tags = upload_analysis["tags"]
                    with st.expander("Analysis Details: planner upload"):
                        st.json(tags.get("extracted_features", {}).get("analysis_details", {}))
                    outfit_name = st.text_input("Item name", value=Path(uploaded.name).stem.replace("_", " ").title(), key=f"planner_upload_name_{target_day}")
                    outfit_category = st.text_input("Category", value=tags.get("category", ""), key=f"planner_upload_category_{target_day}")
                    outfit_subcategory = st.text_input("Clothing type", value=tags.get("subcategory", ""), key=f"planner_upload_subcategory_{target_day}")
                    outfit_color = st.text_input("Primary color", value=tags.get("color", ""), key=f"planner_upload_color_{target_day}")
                    outfit_style = st.text_input("Style", value=tags.get("style", "") or "", key=f"planner_upload_style_{target_day}")
                    outfit_occasions = st.multiselect("Suitable occasions", OCCASIONS + ["Family Gathering", "Festive Casual"], default=tags.get("suitable_occasions", []), key=f"planner_upload_occasions_{target_day}")
                    outfit_market_category = st.selectbox("Garment market category", ["Unknown", "Women's", "Men's", "Unisex"], index=_index(["Unknown", "Women's", "Men's", "Unisex"], tags.get("market_category", "Unknown")), key=f"planner_upload_market_category_{target_day}")
                    if st.button("Add uploaded outfit to this day", key=f"planner_upload_save_{target_day}", use_container_width=True):
                        item = WardrobeItem(
                            name=outfit_name,
                            item_id="",
                            category=outfit_category or "Uncategorized",
                            subcategory=outfit_subcategory or "Unable to determine",
                            market_category=outfit_market_category,
                            color=outfit_color or tags.get("color"),
                            secondary_color=tags.get("secondary_color"),
                            confidence=tags.get("confidence"),
                            style=outfit_style or tags.get("style"),
                            suitable_occasions=outfit_occasions,
                            date_added=date.today().isoformat(),
                            image_name=uploaded.name,
                            image_hash=upload_hash,
                            model_status=tags.get("model_status", "Local vision analysis"),
                            extracted_features=tags.get("extracted_features", {}),
                            classification_uncertain=bool(tags.get("classification_uncertain")),
                        )
                        item = WARDROBE_STORE.add(item)
                        item.image_path = str(WARDROBE_STORE.save_image(item.item_id, uploaded.name, content, IMAGE_DIR))
                        WARDROBE_STORE.update(item)
                        st.session_state.wardrobe_items = WARDROBE_STORE.load()
                        week_days[target_day] = [item.item_id]
                        _planner_save_week(week_start, week_days, week_activities)
                        st.session_state.planner_target_day = None
                        st.session_state.planner_temp_selection.pop(target_day, None)
                        st.rerun()
        if st.button("Close", key=f"planner_close_modal_{target_day}", use_container_width=True):
            st.session_state.planner_target_day = None
            st.session_state.planner_temp_selection.pop(target_day, None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state.wardrobe_items:
        st.info("Upload actual wardrobe images to start planning outfits manually.")