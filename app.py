import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="YOLOv8 Detection", layout="wide")
st.title("🎥 YOLOv8 Detection App")
st.markdown(
    "Upload images or use your webcam for YOLOv8 detection. "
    "Aspect ratio is preserved and display is limited for your screen."
)

# ----------------------------
# Load YOLOv8 model
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ----------------------------
# Helper: Resize for display
# ----------------------------
def resize_for_display(frame, max_width=640, max_height=480):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

# ----------------------------
# Mode selection
# ----------------------------
mode = st.radio("Select Input Mode", ["Upload Image", "Use Webcam"])

# =====================================================
# IMAGE UPLOAD MODE
# =====================================================
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg','jpeg','png'])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        results = model(img)
        annotated = results[0].plot()

        display_frame = resize_for_display(annotated)
        st.image(display_frame, caption="Detected Objects", use_column_width=False)

        # Download button
        _, buffer = cv2.imencode(".png", annotated)
        st.download_button(
            label="Download Annotated Image",
            data=buffer.tobytes(),
            file_name="annotated_image.png",
            mime="image/png"
        )

# =====================================================
# WEBCAM MODE
# =====================================================
elif mode == "Use Webcam":
    stframe = st.empty()
    stop_webcam = st.button("Stop Webcam")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot access webcam. Make sure permissions are allowed.")
    else:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        st.info(f"Webcam resolution: {width}x{height}")

        while True:
            if stop_webcam:
                st.warning("Webcam stopped.")
                break

            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame.")
                break

            results = model(frame)
            annotated = results[0].plot()
            display_frame = resize_for_display(annotated)
            stframe.image(display_frame, channels="RGB", use_column_width=False)

    cap.release()
