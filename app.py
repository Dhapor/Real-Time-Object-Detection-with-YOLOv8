import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from moviepy.editor import VideoFileClip, ImageSequenceClip

# =============== PAGE CONFIG ==================
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")

st.title("🎥 YOLOv8 Object Detection App")
st.markdown(
    "Upload image/video or use your webcam. "
    "Aspect ratio is preserved exactly (portrait stays portrait, landscape stays landscape). "
    "Annotated outputs can be downloaded."
)

# =============== LOAD MODEL ====================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =============== SAFE DISPLAY FUNCTION =========
def fit_to_box(frame, box_width=640, box_height=640):
    """Preserve aspect ratio, shrink ONLY if larger."""
    h, w = frame.shape[:2]
    scale = min(box_width / w, box_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))

# =============== ROTATION/MIRROR FIX ===========
def fix_frame_orientation(frame, flip=False):
    """Flip frame horizontally if webcam mirrored, or rotate if needed."""
    if flip:
        frame = cv2.flip(frame, 1)
    return frame

# =============== MODE SELECT ===================
mode = st.radio("Select Input Mode", ["Upload Image/Video", "Use Webcam"])

# =====================================================
# UPLOAD MODE
# =====================================================
if mode == "Upload Image/Video":
    file = st.file_uploader("Upload", type=["jpg", "jpeg", "png", "mp4", "mov"])

    if file:
        file_type = file.type

        # ---------------- IMAGE ----------------
        if "image" in file_type:
            bytes_data = file.read()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            results = model(img)
            annotated = results[0].plot()

            display = fit_to_box(annotated)
            st.image(display, caption="Detected", channels="RGB")

            _, buf = cv2.imencode(".png", annotated)
            st.download_button("Download Annotated Image",
                               data=buf.tobytes(),
                               file_name="annotated.png")

        # ---------------- VIDEO ----------------
        elif "video" in file_type:
            temp_input = tempfile.NamedTemporaryFile(delete=False)
            temp_input.write(file.read())
            temp_input.close()

            cap = cv2.VideoCapture(temp_input.name)

            progress = st.progress(0)
            stframe = st.empty()
            frames_out = []
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stop = st.button("Stop Video Processing")

            count = 0
            while cap.isOpened():
                if stop:
                    st.warning("Stopped by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # --- Fix orientation (important for portrait iPhone videos) ---
                frame = fix_frame_orientation(frame)

                results = model(frame)
                annotated = results[0].plot()

                frames_out.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                display = fit_to_box(annotated)
                stframe.image(display, channels="RGB")

                count += 1
                progress.progress(min(count / total, 1.0))

            cap.release()
            progress.empty()

            if frames_out:
                temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                clip = ImageSequenceClip(frames_out, fps=30)
                clip.write_videofile(temp_out.name, codec="libx264")

                st.download_button(
                    "Download Annotated Video",
                    data=open(temp_out.name, "rb").read(),
                    file_name="annotated_video.mp4"
                )

# =====================================================
# WEBCAM MODE
# =====================================================
else:
    stframe = st.empty()
    stop = st.button("Stop Webcam")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam.")
    else:
        width = int(cap.get(3))
        height = int(cap.get(4))
        st.info(f"Webcam resolution: {width}x{height}")

        while True:
            if stop:
                st.warning("Webcam stopped")
                break

            ret, frame = cap.read()
            if not ret:
                st.warning("Frame not received.")
                break

            # --- Fix webcam mirror ---
            frame = fix_frame_orientation(frame, flip=True)

            results = model(frame)
            annotated = results[0].plot()

            display = fit_to_box(annotated)
            stframe.image(display, channels="RGB")

    cap.release()
