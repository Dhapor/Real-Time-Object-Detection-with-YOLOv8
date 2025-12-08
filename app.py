import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from moviepy.editor import VideoFileClip, ImageSequenceClip
import os

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    layout="wide"
)

st.title("🎥 YOLOv8 Object Detection App")
st.markdown(
    "Upload an image/video or use your webcam to detect objects in real-time using YOLOv8. "
    "Frames maintain their aspect ratio and fit on screen. You can download annotated results."
)

# ----------------------------
# Load YOLOv8 model
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ----------------------------
# Display scaling function
# ----------------------------
def resize_for_display(frame, max_width=640, max_height=480):
    h, w = frame.shape[:2]
    scale = min(max_width/w, max_height/h, 1.0)  # only shrink if larger
    new_w, new_h = int(w*scale), int(h*scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

# ----------------------------
# Select mode
# ----------------------------
mode = st.radio("Select Input Mode", ["Upload Image/Video", "Use Webcam"])

# =====================================================
# UPLOAD IMAGE/VIDEO MODE
# =====================================================
if mode == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg','jpeg','png','mp4','mov'])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        # --- IMAGE ---
        if "image" in file_type:
            file_bytes = uploaded_file.read()
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            results = model(img)
            annotated = results[0].plot()
            
            display_frame = resize_for_display(annotated)
            st.image(display_frame, caption="Detected Objects", use_container_width=False)

            # Download
            _, buffer = cv2.imencode(".png", annotated)
            st.download_button(
                label="Download Annotated Image",
                data=buffer.tobytes(),
                file_name="annotated_image.png",
                mime="image/png"
            )

        # --- VIDEO ---
        elif "video" in file_type:
            temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mov" if "quicktime" in file_type else ".mp4")
            temp_in.write(uploaded_file.read())
            temp_in.close()

            # Convert MOV → MP4 if needed
            if "quicktime" in file_type:
                temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                clip = VideoFileClip(temp_in.name)
                clip.write_videofile(temp_out.name, codec='libx264')
                video_path = temp_out.name
            else:
                video_path = temp_in.name

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)

            current_frame = 0
            stop_processing = st.button("Stop Video Processing")
            annotated_frames = []

            while cap.isOpened():
                if stop_processing:
                    st.warning("Video processing stopped.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated = results[0].plot()
                annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                # Resize for display only
                display_frame = resize_for_display(annotated)
                stframe.image(display_frame, channels="RGB", use_container_width=False)

                current_frame += 1
                progress_bar.progress(min(current_frame / total_frames, 1.0))

            cap.release()
            progress_bar.empty()

            if annotated_frames:
                # Save annotated video
                temp_video_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                clip = ImageSequenceClip(annotated_frames, fps=30)
                clip.write_videofile(temp_video_out.name, codec="libx264")
                st.download_button(
                    label="Download Annotated Video",
                    data=open(temp_video_out.name, "rb").read(),
                    file_name="annotated_video.mp4",
                    mime="video/mp4"
                )

# =====================================================
# LIVE WEBCAM MODE
# =====================================================
elif mode == "Use Webcam":
    stframe = st.empty()
    stop_webcam = st.button("Stop Webcam")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open webcam. Make sure no other app is using it and camera permissions are allowed.")
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
                st.warning("Failed to grab frame from webcam.")
                break

            results = model(frame)
            annotated = results[0].plot()

            display_frame = resize_for_display(annotated)
            stframe.image(display_frame, channels="RGB", use_container_width=False)

    cap.release()
