import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# Setting up the page configuration
st.set_page_config(page_title="YOLOv8 Object Detection App")
st.title("YOLOv8 Object Detection App 🚗🧍🏽‍♂️🚌")
st.markdown("Choose input mode to detect objects using YOLOv8:")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Lightweight model for speed

# Load the model
model = load_model()

# User input mode: Upload image/video or use webcam
mode = st.radio("Select Input Mode", ["Upload Image/Video", "Use Webcam"])

# If the user uploads an image/video
if mode == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4'])

    # If the user uploads an image
    if uploaded_file is not None:
        file_type = uploaded_file.type

        if 'image' in file_type:
            file_bytes = uploaded_file.read()
            np_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Run object detection on the image
            results = model(np_img)
            annotated = results[0].plot()

            # Display the annotated image
            st.image(annotated, caption="Detected Image", use_container_width=True)

        # If the user uploads a video
        elif 'video' in file_type:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                annotated_frame = results[0].plot()

                # Display each annotated frame in the video stream
                stframe.image(annotated_frame, channels="BGR")
            cap.release()

# If the user chooses webcam input
elif mode == "Use Webcam":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Run object detection on the webcam frame
            results = model(img)
            annotated = results[0].plot()

            # Return the annotated frame
            return annotated

    # Start the webcam stream with object detection
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)
