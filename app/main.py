import hashlib
from pathlib import Path

import streamlit as st
import pandas as pd

from core.schemas import UserProfile, WardrobeItem
from core.wardrobe_store import WardrobeStore
from features.body_shape import analyze_body_shape
from features.clothing_detection import detect_and_classify_clothing, detect_clothing
from features.digital_wardrobe import get_fashion_embedding, segment_clothing_items, segment_and_tag_clothing
from features.virtual_try_on import generate_virtual_try_on
from features.face_shape import analyze_face_shape
from features.skin_tone import analyze_skin_tone
from features.planner import generate_weekly_plan, regenerate_day
from features.recommendation import recommend_outfits


FACE_LANDMARKER_MODEL = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
POSE_LANDMARKER_MODEL = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"
SAM2_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "sam2_hiera_small.pt"
YOLO11_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "yolo11n.pt"
WARDROBE_STORE = WardrobeStore(Path(__file__).resolve().parents[1] / "data" / "wardrobe" / "items.json")
WARDROBE_IMAGE_DIR = Path(__file__).resolve().parents[1] / "data" / "wardrobe" / "images"


def _image_signature(image_bytes: bytes | None) -> str | None:
    if not image_bytes:
        return None
    return hashlib.sha256(image_bytes).hexdigest()


def _option_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else len(options) - 1


st.set_page_config(
    page_title="AI Fashion Stylist",
    page_icon="FS",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #1d2421;
        --muted: #68726d;
        --paper: #f5f4ef;
        --panel: #fffef9;
        --line: #d9ddd5;
        --coral: #de684e;
        --teal: #2e6d67;
        --yellow: #e8c56a;
    }

    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    [data-testid="stSidebar"] { background: #e6ece5; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.25rem; }
    .brand { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.04em; margin-bottom: 0.2rem; }
    .brand-mark { color: var(--coral); }
    .eyebrow { color: var(--teal); font: 500 0.72rem 'DM Mono', monospace; letter-spacing: 0.11em; text-transform: uppercase; }
    .hero { padding: 1.1rem 0 1.8rem; border-bottom: 1px solid var(--line); }
    .hero h1 { font-size: clamp(2.4rem, 5vw, 4.65rem); line-height: 0.98; letter-spacing: -0.075em; max-width: 720px; margin: 0.55rem 0 1rem; }
    .hero p { color: var(--muted); font-size: 1rem; max-width: 560px; line-height: 1.6; }
    .section-title { font-size: 1.15rem; font-weight: 800; letter-spacing: -0.04em; margin: 1.7rem 0 0.75rem; }
    .metric { background: var(--panel); border: 1px solid var(--line); padding: 1rem; min-height: 100px; }
    .metric-value { font-size: 1.85rem; font-weight: 800; letter-spacing: -0.06em; }
    .metric-label { color: var(--muted); font-size: 0.76rem; margin-top: 0.25rem; }
    .feature { background: var(--panel); border-top: 3px solid var(--teal); border-bottom: 1px solid var(--line); padding: 1rem; min-height: 155px; }
    .feature-number { color: var(--coral); font: 500 0.73rem 'DM Mono', monospace; }
    .feature h3 { font-size: 1rem; margin: 0.55rem 0 0.35rem; letter-spacing: -0.03em; }
    .feature p { color: var(--muted); font-size: 0.82rem; line-height: 1.5; margin: 0; }
    .recommendation { background: var(--teal); color: #f8f8f1; padding: 1.3rem; min-height: 180px; }
    .recommendation h3 { font-size: 1.1rem; margin: 0.45rem 0; }
    .recommendation p { color: #d9e8df; font-size: 0.85rem; line-height: 1.55; }
    .status { display: inline-block; border: 1px solid #aebbb0; padding: 0.25rem 0.45rem; color: var(--teal); font: 500 0.68rem 'DM Mono', monospace; text-transform: uppercase; }
    div[data-testid="stFileUploader"] { background: var(--panel); border: 1px dashed #aebbb0; padding: 0.45rem; }
    .stButton > button { border-radius: 0; border: 1px solid var(--ink); background: var(--ink); color: white; font-weight: 700; padding: 0.65rem 1rem; }
    .stButton > button:hover { background: var(--coral); border-color: var(--coral); color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


if "profile" not in st.session_state:
    st.session_state.profile = UserProfile()
if "wardrobe_items" not in st.session_state:
    st.session_state.wardrobe_items = WARDROBE_STORE.load()

profile: UserProfile = st.session_state.profile
profile.wardrobe_items = st.session_state.wardrobe_items

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">/</span> style intelligence</div>', unsafe_allow_html=True)
    st.caption("A personal studio for better outfit decisions.")
    st.divider()
    st.markdown('<div class="eyebrow">Workspace</div>', unsafe_allow_html=True)
    view = st.radio("Navigate", ["Overview", "My wardrobe", "Clothing scan", "Virtual try-on", "Weekly planner"], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="eyebrow">Profile status</div>', unsafe_allow_html=True)
    st.markdown(f"**{profile.analysis_count}/3** visual signals ready")
    st.progress(profile.analysis_count / 3)
    st.caption("Upload a clear portrait and full-body image to unlock personalized analysis.")


st.markdown(
    '<div class="hero"><div class="eyebrow">AI-FASHION-STYLIST / 01</div><h1>Get dressed with a little more intention.</h1><p>Build a visual profile, understand what is already in your closet, and get outfit ideas that feel like you.</p></div>',
    unsafe_allow_html=True,
)

if view == "Overview":
    st.markdown('<div class="section-title">Start with your profile</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="eyebrow">SKIN TONE / FACE</div>', unsafe_allow_html=True)
        portrait = st.file_uploader("Choose a portrait file", type=["jpg", "jpeg", "png"], key="portrait")
        camera_portrait = st.camera_input("Or use camera for skin tone", key="camera_portrait")
        st.markdown('<div class="eyebrow" style="margin-top:1rem">BODY SHAPE</div>', unsafe_allow_html=True)
        full_body = st.file_uploader("Choose a full-body file", type=["jpg", "jpeg", "png"], key="full_body")
        camera_full_body = st.camera_input("Or use camera for full body", key="camera_full_body")
        wardrobe_uploads = st.file_uploader("Add wardrobe pieces", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="wardrobe")
        profile_image = camera_portrait or portrait
        portrait_bytes = profile_image.getvalue() if profile_image else None
        body_image = camera_full_body or full_body
        body_bytes = body_image.getvalue() if body_image else None
        portrait_signature = _image_signature(portrait_bytes)
        body_signature = _image_signature(body_bytes)
        new_portrait = portrait_signature and portrait_signature != st.session_state.get("last_portrait_signature")
        new_body = body_signature and body_signature != st.session_state.get("last_body_signature")

        if new_portrait:
            profile.face_shape = None
            profile.skin_tone = None
            st.session_state.pop("face_shape_details", None)
            st.session_state.pop("skin_tone_details", None)
        if new_body:
            profile.body_shape = None
            st.session_state.pop("body_shape_details", None)

        analyze_clicked = st.button("Analyze current photos", use_container_width=True)
        if analyze_clicked or new_portrait or new_body:
            if portrait_bytes and (analyze_clicked or new_portrait):
                try:
                    face_result = analyze_face_shape(portrait_bytes, FACE_LANDMARKER_MODEL)
                    profile.face_shape = face_result.shape
                    st.session_state.face_shape_details = face_result
                except FileNotFoundError:
                    profile.face_shape = "Model checkpoint needed"
                    st.info("Add models/face_landmarker.task to enable face-shape analysis.")
                except ImportError:
                    profile.face_shape = "Analysis dependencies needed"
                    st.warning("Install MediaPipe to run face-shape analysis.")
                except ValueError as error:
                    profile.face_shape = "Image needs review"
                    st.warning(str(error))
                try:
                    tone_result = analyze_skin_tone(portrait_bytes, FACE_LANDMARKER_MODEL)
                    profile.skin_tone = tone_result.label
                    st.session_state.skin_tone_details = tone_result
                except FileNotFoundError:
                    profile.skin_tone = "Model checkpoint needed"
                    st.info("Add models/face_landmarker.task to enable landmark-based skin-tone analysis.")
                except ImportError:
                    profile.skin_tone = "Analysis dependencies needed"
                    st.warning("Install MediaPipe, OpenCV, and scikit-learn to run skin-tone analysis.")
                except ValueError as error:
                    profile.skin_tone = "Image needs review"
                    st.warning(str(error))
            if body_bytes and (analyze_clicked or new_body):
                try:
                    body_result = analyze_body_shape(body_bytes, POSE_LANDMARKER_MODEL)
                    profile.body_shape = body_result.shape
                    st.session_state.body_shape_details = body_result
                except FileNotFoundError:
                    profile.body_shape = "Model checkpoint needed"
                    st.info("Add models/pose_landmarker_lite.task to enable body-shape analysis.")
                except ImportError:
                    profile.body_shape = "Analysis dependencies needed"
                    st.warning("Install MediaPipe to run body-shape analysis.")
                except ValueError as error:
                    profile.body_shape = "Needs a clearer full-body image"
                    st.warning(f"Body analysis: {error} Try a standing, front-facing photo with the whole body visible and arms slightly away from the torso.")
            if wardrobe_uploads:
                for file in wardrobe_uploads:
                    item = WardrobeItem(name=file.name, image_name=file.name, model_status="Awaiting SAM 2 + FashionCLIP")
                    saved_item = WARDROBE_STORE.add(item)
                    st.session_state.wardrobe_items.append(saved_item)
            if portrait_signature:
                st.session_state.last_portrait_signature = portrait_signature
            if body_signature:
                st.session_state.last_body_signature = body_signature
            st.session_state.profile = profile
            st.success("Current photos analyzed. Results above belong to the latest captured or uploaded images.")
    with right:
        st.markdown('<div class="eyebrow">Current signals</div>', unsafe_allow_html=True)
        signals = [("Face shape", profile.face_shape or "Waiting for photo"), ("Skin tone", profile.skin_tone or "Waiting for photo"), ("Body shape", profile.body_shape or "Waiting for full-body photo")]
        for label, value in signals:
            st.markdown(f'<div class="metric" style="margin-bottom:0.55rem"><div class="metric-label">{label}</div><div style="font-weight:700">{value}</div></div>', unsafe_allow_html=True)
        face_details = st.session_state.get("face_shape_details")
        if face_details:
            st.caption(f"Face box: {face_details.face_box} · confidence {face_details.confidence:.0%}")
        body_details = st.session_state.get("body_shape_details")
        if body_details:
            detail = getattr(body_details, "detail", "Detailed proportions will appear after re-analysis.")
            st.markdown(f"**{detail}**")
            for feature in getattr(body_details, "features", ("Re-run analysis to see silhouette features.",)):
                st.markdown(f"- {feature}")
        tone_details = st.session_state.get("skin_tone_details")
        if tone_details:
            shade_level = getattr(tone_details, "shade_level", tone_details.label)
            undertone = getattr(tone_details, "undertone", "Pending re-analysis")
            undertone_detail = getattr(tone_details, "undertone_detail", "run analysis again for LAB undertone detail")
            description = getattr(tone_details, "description", f"Your complexion reads as {shade_level} with a {undertone.lower()} balance.")
            st.markdown(f"**{description}**")
            st.caption(f"Undertone detail: {undertone_detail}")
            guidance = getattr(tone_details, "color_guidance", ())
            if guidance:
                st.caption("Colors to explore: " + " · ".join(guidance))

    st.markdown('<div class="section-title">Choose the moment</div>', unsafe_allow_html=True)
    profile.occasion = st.selectbox("Occasion", ["Everyday", "Work", "Date night", "Wedding guest", "Travel", "Weekend"], label_visibility="collapsed")
    profile.style_preferences = st.multiselect("Style direction", ["Minimal", "Relaxed", "Polished", "Romantic", "Bold", "Classic"], default=profile.style_preferences, label_visibility="collapsed")

    st.markdown('<div class="section-title">Your first read</div>', unsafe_allow_html=True)
    metric_a, metric_b, metric_c = st.columns(3)
    for container, value, label in [(metric_a, len(profile.wardrobe_items), "pieces in wardrobe"), (metric_b, profile.analysis_count, "signals ready"), (metric_c, "--", "style score")]:
        with container:
            st.markdown(f'<div class="metric"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">What is powering the studio</div>', unsafe_allow_html=True)
    features = [
        ("01", "Visual profile", "Face shape, skin tone, and body proportions form a useful starting point."),
        ("02", "Closet intelligence", "Your own pieces become searchable by category, color, and visual similarity."),
        ("03", "Outfit direction", "Occasion, personal taste, and trend context meet in one recommendation."),
    ]
    feature_columns = st.columns(3, gap="medium")
    for column, (number, title, text) in zip(feature_columns, features):
        with column:
            st.markdown(f'<div class="feature"><div class="feature-number">{number} / MODULE</div><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">A starting direction</div>', unsafe_allow_html=True)
    rec_left, rec_right = st.columns([1.25, 0.75], gap="medium")
    recommendations = recommend_outfits(profile, st.session_state.wardrobe_items, profile.occasion, top_k=1)
    with rec_left:
        if recommendations:
            recommendation = recommendations[0]
            item_names = " + ".join(item.name for item in recommendation.items)
            reasons = " ".join(recommendation.reasons)
            st.markdown(f'<div class="recommendation"><div class="eyebrow" style="color:#e8c56a">{profile.occasion.upper()} EDIT</div><h3>{item_names}</h3><p>Score {recommendation.score:.0%}. {reasons}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="recommendation"><div class="eyebrow" style="color:#e8c56a">{profile.occasion.upper()} EDIT</div><h3>Your wardrobe needs more detail.</h3><p>Add categorized pieces to generate a transparent outfit recommendation.</p></div>', unsafe_allow_html=True)
    with rec_right:
        st.markdown('<div class="feature"><span class="status">Pipeline staged</span><h3>Next signal</h3><p>Add the Face and Pose Landmarker checkpoints to activate the complete visual profile.</p></div>', unsafe_allow_html=True)

elif view == "My wardrobe":
    st.markdown('<div class="eyebrow">WARDROBE / 02</div><h2>Everything you already own.</h2>', unsafe_allow_html=True)
    st.caption("Upload pieces now, edit their tags below, and connect SAM 2/FashionCLIP checkpoints when ready.")
    wardrobe_file = st.file_uploader("Add a wardrobe piece", type=["jpg", "jpeg", "png"], key="wardrobe_page_upload")
    if wardrobe_file and st.button("Add to wardrobe", use_container_width=True):
        image_bytes = wardrobe_file.getvalue()
        try:
            detection = detect_clothing(image_bytes, YOLO11_CHECKPOINT)
            if not detection.detections:
                st.warning("No objects were detected. Use a clearer clothing image or a fashion-trained YOLO checkpoint.")
            else:
                segmented_items = segment_clothing_items(image_bytes, detection.detections)
                added = 0
                for index, segmented in enumerate(segmented_items, start=1):
                    item_hash = _image_signature(Path(segmented.crop_path).read_bytes())
                    duplicate = next((stored for stored in st.session_state.wardrobe_items if stored.image_hash == item_hash), None)
                    if duplicate:
                        continue
                    item = WardrobeItem(
                        name=f"{Path(wardrobe_file.name).stem} {index}",
                        image_name=Path(segmented.crop_path).name,
                        image_path=segmented.crop_path,
                        image_hash=item_hash,
                        category=segmented.label,
                        color=segmented.color,
                        pattern=segmented.pattern,
                        embedding=segmented.embedding,
                        confidence=segmented.confidence,
                        mask_path=segmented.mask_path,
                        model_status=segmented.model_status,
                    )
                    saved_item = WARDROBE_STORE.add(item)
                    st.session_state.wardrobe_items.append(saved_item)
                    added += 1
                st.success(f"Added {added} detected wardrobe item(s).")
                st.rerun()
        except (FileNotFoundError, ImportError, TypeError, ValueError, RuntimeError) as error:
            st.warning(str(error))
    if st.session_state.wardrobe_items:
        for item in st.session_state.wardrobe_items:
            with st.expander(item.name):
                if item.image_path and Path(item.image_path).exists():
                    st.image(item.image_path, caption=item.name, use_container_width=True)
                st.caption(item.model_status or "No model status")
                category = st.selectbox("Category", ["Top", "Bottom", "Dress", "Outerwear", "Shoes", "Accessory", "Uncategorized"], index=_option_index(["Top", "Bottom", "Dress", "Outerwear", "Shoes", "Accessory", "Uncategorized"], item.category), key=f"category_{item.item_id}")
                color = st.text_input("Color", value=item.color or "", key=f"color_{item.item_id}")
                pattern = st.text_input("Pattern", value=item.pattern or "", key=f"pattern_{item.item_id}")
                edit_col, delete_col = st.columns(2)
                with edit_col:
                    if st.button("Save tags", key=f"save_{item.item_id}", use_container_width=True):
                        item.category, item.color, item.pattern = category, color or None, pattern or None
                        item.model_status = item.model_status or "Manual tags"
                        WARDROBE_STORE.update(item)
                        st.success("Wardrobe tags saved.")
                with delete_col:
                    if st.button("Delete", key=f"delete_{item.item_id}", use_container_width=True):
                        WARDROBE_STORE.delete(item.item_id)
                        st.session_state.wardrobe_items = [stored for stored in st.session_state.wardrobe_items if stored.item_id != item.item_id]
                        st.rerun()
    else:
        st.info("Your wardrobe is empty. Add a clothing image to begin.")

elif view == "Clothing scan":
    st.markdown('<div class="eyebrow">CLOTHING DETECTION / 04</div><h2>Read an outfit at a glance.</h2>', unsafe_allow_html=True)
    st.caption("YOLO detects the classes supported by the configured checkpoint. FashionCLIP is not used by this screen.")
    scan_file = st.file_uploader("Upload an outfit image", type=["jpg", "jpeg", "png"], key="clothing_scan")
    if scan_file and st.button("Detect clothing", use_container_width=True):
        try:
            detection = detect_and_classify_clothing(scan_file.getvalue(), YOLO11_CHECKPOINT)
            st.session_state.clothing_detection = detection
            st.success(f"Detected {len(detection.detections)} objects above the confidence threshold.")
        except FileNotFoundError:
            st.info("Add models/yolo11n.pt to enable YOLO11 detection.")
        except (ImportError, TypeError, ValueError, RuntimeError) as error:
            st.warning(str(error))
    detection = st.session_state.get("clothing_detection")
    if detection:
        st.image(detection.annotated_image, caption="YOLO annotated output", use_container_width=True)
        st.caption(f"Device: {detection.device} · Confidence threshold: {detection.confidence_threshold:.2f}")
        if detection.model_classes:
            st.info(f"{detection.model_status} Classes: {', '.join(detection.model_classes)}")
        if not detection.detections:
            st.info("No objects were detected above the confidence threshold.")
        rows = [
            {
                "Item": item.category,
                "Label": item.label,
                "Confidence": f"{item.confidence:.1%}",
                "X1": item.bbox[0],
                "Y1": item.bbox[1],
                "X2": item.bbox[2],
                "Y2": item.bbox[3],
            }
            for item in detection.detections
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for index, detected_item in enumerate(detection.detections, start=1):
            st.markdown(f'<div class="metric" style="margin:0.5rem 0"><b>{index}. {detected_item.category}</b><br><span class="metric-label">confidence {detected_item.confidence:.0%} · box {detected_item.box}</span></div>', unsafe_allow_html=True)

elif view == "Virtual try-on":
    st.markdown('<div class="eyebrow">VIRTUAL TRY-ON / 05</div><h2>See the piece on your body.</h2>', unsafe_allow_html=True)
    st.caption("Upload a full-body photo and a dress or clothing-item photo. The generator creates a visual preview while preserving the overall person and pose.")
    person_file = st.file_uploader("Your full-body photo", type=["jpg", "jpeg", "png"], key="try_on_person")
    garment_file = st.file_uploader("Dress or clothing item", type=["jpg", "jpeg", "png"], key="try_on_garment")
    if st.button("Generate try-on preview", use_container_width=True):
        if not person_file or not garment_file:
            st.warning("Add both images before generating a preview.")
        else:
            with st.spinner("Generating your outfit preview..."):
                try:
                    generated_path = generate_virtual_try_on(person_file.getvalue(), garment_file.getvalue())
                    st.session_state.try_on_path = str(generated_path)
                except ImportError as error:
                    st.warning(str(error))
                except Exception as error:
                    st.error(f"Try-on generation failed: {error}")
    if st.session_state.get("try_on_path"):
        st.image(st.session_state.try_on_path, caption="AI-generated try-on preview", use_container_width=True)
        st.info("This is a visual approximation. It may change garment details, fit, face, or body proportions; use it for styling exploration, not exact fit measurement.")

else:
    st.markdown('<div class="eyebrow">PLANNER / 03</div><h2>A week that gets easier to dress for.</h2>', unsafe_allow_html=True)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    default_occasions = ["Work", "Work", "Work", "Work", "Work", "Weekend", "Weekend"]
    occasions = []
    planner_columns = st.columns(7, gap="small")
    for index, (column, day) in enumerate(zip(planner_columns, days)):
        with column:
            occasion = st.selectbox("Occasion", ["Everyday", "Work", "Date night", "Wedding guest", "Travel", "Weekend"], index=_option_index(["Everyday", "Work", "Date night", "Wedding guest", "Travel", "Weekend"], default_occasions[index]), key=f"planner_occasion_{day}", label_visibility="collapsed")
            occasions.append(occasion)
    if st.button("Generate weekly plan", use_container_width=True):
        st.session_state.weekly_plan = generate_weekly_plan(profile, st.session_state.wardrobe_items, occasions, days)
    weekly_plan = st.session_state.get("weekly_plan")
    if weekly_plan:
        plan_columns = st.columns(7, gap="small")
        for index, (column, planned_day) in enumerate(zip(plan_columns, weekly_plan.days)):
            with column:
                outfit_text = " + ".join(item.name for item in planned_day.outfit.items) if planned_day.outfit else "Not planned"
                st.markdown(f'<div class="metric"><div class="eyebrow">{planned_day.day}</div><strong>{outfit_text}</strong><p style="color:var(--muted);font-size:0.78rem">{planned_day.occasion}</p><p style="color:var(--muted);font-size:0.78rem">{planned_day.reason}</p></div>', unsafe_allow_html=True)
                if st.button("Regenerate", key=f"regenerate_{planned_day.day}", use_container_width=True):
                    st.session_state.weekly_plan = regenerate_day(weekly_plan, index, profile, st.session_state.wardrobe_items)
                    st.rerun()
    else:
        st.info("Generate a week from the pieces currently stored in your wardrobe.")
