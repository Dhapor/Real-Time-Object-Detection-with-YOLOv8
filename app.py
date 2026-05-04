import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from moviepy.editor import VideoFileClip, ImageSequenceClip

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }

  .hero-title { font-size: 2.8rem; font-weight: 700; color: #e65100; text-align: center; margin-bottom: 0; }
  .hero-sub   { font-size: 1.05rem; color: #888; text-align: center; margin-top: 4px; }
  .section-header {
    font-size: 1.4rem; font-weight: 600; color: #e65100;
    border-bottom: 2px solid #e65100; padding-bottom: 6px; margin-top: 1.8rem;
  }
  .class-chip {
    display: inline-block; background: #fff3e0; color: #e65100;
    border: 1px solid #e65100; border-radius: 20px;
    padding: 3px 10px; margin: 3px; font-size: 0.82rem; font-weight: 600;
  }
  .stat-card {
    background: #e65100; color: white; border-radius: 10px;
    padding: 18px; text-align: center;
  }
  .stat-card .value { font-size: 1.8rem; font-weight: 700; }
  .stat-card .label { font-size: 0.85rem; opacity: 0.85; margin-top: 2px; }
  .info-card {
    background: #fff8f5; border-left: 4px solid #e65100;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 10px;
  }
  .info-card h4 { color: #e65100; margin: 0 0 4px 0; font-size: 1rem; }
  .info-card p  { color: #555; margin: 0; font-size: 0.9rem; }
  hr.divider { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# COCO classes YOLOv8n detects
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def fit_to_box(frame, box_width=640, box_height=640):
    h, w = frame.shape[:2]
    scale = min(box_width / w, box_height / h, 1.0)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

def fix_frame_orientation(frame, flip=False):
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🎯 YOLOv8 Object Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Real-time object detection in images and videos &nbsp;|&nbsp; Built by Datapsalm</p>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

tab_home, tab_detect, tab_about = st.tabs(["🏠 Overview", "📸 Detect Objects", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    col_left, col_right = st.columns([1.4, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-header">What Is This App?</p>', unsafe_allow_html=True)
        st.markdown("""
This app uses **YOLOv8** (You Only Look Once, version 8) — one of the fastest and most accurate
object detection models available — to identify and locate objects in images, videos, and live webcam feeds.

Upload a photo or video on the **Detect Objects** tab to see bounding boxes drawn around every
detected object, along with its label and confidence score. You can also download the annotated result.
        """)

        st.markdown('<br>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="stat-card"><div class="value">80</div><div class="label">Object Classes</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="stat-card"><div class="value">YOLOv8n</div><div class="label">Model Variant</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stat-card"><div class="value">Real-time</div><div class="label">Detection Speed</div></div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<p class="section-header">Use Cases</p>', unsafe_allow_html=True)
        for title, desc in [
            ("🏙️ Smart City", "Count pedestrians, vehicles, and cyclists at intersections"),
            ("🏪 Retail Analytics", "Track product placement and customer movement in stores"),
            ("🔒 Security", "Detect people, bags, or vehicles in surveillance footage"),
            ("🚗 Autonomous Vehicles", "Identify road objects for self-driving navigation"),
            ("🏭 Industrial QA", "Spot defects or misplaced items on production lines"),
        ]:
            st.markdown(f"""
            <div class="info-card">
              <h4>{title}</h4>
              <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">What Can It Detect?</p>', unsafe_allow_html=True)
    st.markdown("YOLOv8 is trained on the **COCO dataset** and can detect 80 everyday object classes:")
    st.markdown('<br>', unsafe_allow_html=True)
    chips_html = " ".join(f'<span class="class-chip">{c}</span>' for c in COCO_CLASSES)
    st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">How YOLO Works</p>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, (step, label) in zip(cols, [
        ("1️⃣", "The image is divided into a grid of cells"),
        ("2️⃣", "Each cell predicts bounding boxes and class probabilities simultaneously"),
        ("3️⃣", "Non-max suppression removes overlapping duplicate detections"),
        ("4️⃣", "Final boxes are drawn with labels and confidence scores"),
    ]):
        with col:
            st.info(f"**{step}** {label}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DETECT
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.markdown('<p class="section-header">Run Detection</p>', unsafe_allow_html=True)

    conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05,
                               help="Only show detections above this confidence level. Lower = more detections, higher = fewer but more certain.")

    mode = st.radio("Select input mode", ["Upload Image/Video", "Use Webcam"], horizontal=True)

    if mode == "Upload Image/Video":
        file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])

        if file:
            file_type = file.type

            if "image" in file_type:
                bytes_data = file.read()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                results = model(img, conf=conf_threshold)
                annotated = results[0].plot()

                boxes = results[0].boxes
                detected_classes = [model.names[int(c)] for c in boxes.cls] if boxes is not None else []

                col_img, col_stats = st.columns([2, 1])
                with col_img:
                    display = fit_to_box(annotated)
                    st.image(display, caption="Detection result", channels="BGR")
                    _, buf = cv2.imencode(".png", annotated)
                    st.download_button("Download annotated image", data=buf.tobytes(), file_name="detected.png")

                with col_stats:
                    st.markdown("**Detection summary**")
                    st.metric("Objects found", len(detected_classes))
                    if detected_classes:
                        from collections import Counter
                        counts = Counter(detected_classes)
                        for cls, cnt in counts.most_common():
                            st.write(f"- {cls}: **{cnt}**")

            elif "video" in file_type:
                temp_input = tempfile.NamedTemporaryFile(delete=False)
                temp_input.write(file.read())
                temp_input.close()

                cap = cv2.VideoCapture(temp_input.name)
                progress = st.progress(0)
                stframe  = st.empty()
                frames_out = []
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                stop  = st.button("Stop processing")

                count = 0
                while cap.isOpened():
                    if stop:
                        st.warning("Stopped by user.")
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = fix_frame_orientation(frame)
                    results = model(frame, conf=conf_threshold)
                    annotated = results[0].plot()
                    frames_out.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                    stframe.image(fit_to_box(annotated), channels="BGR")
                    count += 1
                    progress.progress(min(count / total, 1.0))

                cap.release()
                progress.empty()

                if frames_out:
                    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    clip = ImageSequenceClip(frames_out, fps=30)
                    clip.write_videofile(temp_out.name, codec="libx264")
                    st.download_button("Download annotated video",
                                       data=open(temp_out.name, "rb").read(),
                                       file_name="detected_video.mp4")
    else:
        st.info("Webcam detection runs locally in your browser session. Press Stop to end the stream.")
        stframe = st.empty()
        stop    = st.button("Stop Webcam")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam.")
        else:
            w = int(cap.get(3))
            h = int(cap.get(4))
            st.caption(f"Webcam resolution: {w}x{h}")
            while True:
                if stop:
                    st.warning("Webcam stopped.")
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frame = fix_frame_orientation(frame, flip=True)
                results = model(frame, conf=conf_threshold)
                annotated = results[0].plot()
                stframe.image(fit_to_box(annotated), channels="BGR")
        cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<p class="section-header">About YOLOv8</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
**YOLO** (You Only Look Once) is a family of real-time object detection models developed
by Ultralytics. Unlike older two-stage detectors, YOLO processes the entire image in a single
forward pass — making it extremely fast without sacrificing accuracy.

**YOLOv8n** (nano) is the smallest variant, optimized for speed. It's ideal for real-time
applications where low latency matters more than maximum precision.
        """)
        for label, value in [
            ("Model", "YOLOv8n (nano)"),
            ("Framework", "Ultralytics / PyTorch"),
            ("Training dataset", "COCO (80 classes, 330K images)"),
            ("Input size", "640 × 640 px"),
            ("Developer", "Ultralytics"),
        ]:
            st.markdown(f"**{label}:** {value}")

    with col2:
        st.markdown("""
**Understanding confidence scores:**

| Score | Meaning |
|---|---|
| 0.9+ | Very high confidence — almost certain |
| 0.7 to 0.9 | High confidence |
| 0.5 to 0.7 | Moderate confidence |
| 0.25 to 0.5 | Low confidence — may be noisy |
| < 0.25 | Filtered out (below default threshold) |

Use the confidence slider on the Detect tab to adjust the minimum threshold.
        """)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Credits</p>', unsafe_allow_html=True)
    st.markdown("Built by **Datapsalm** using [Ultralytics YOLOv8](https://docs.ultralytics.com), OpenCV, and Streamlit.")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### YOLOv8 Detector")
    st.markdown("Upload an image or video to detect objects in real-time.")
    st.markdown("---")
    st.markdown("**Model:** YOLOv8n")
    st.markdown("**Classes:** 80 (COCO)")
    st.markdown("**Input:** Image, Video, Webcam")
    st.markdown("---")
    st.caption("Built by Datapsalm")
