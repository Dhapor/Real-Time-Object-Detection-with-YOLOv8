import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

st.set_page_config(page_title="YOLOv8 Object Detection App")
st.title("YOLOv8 Object Detection App 🚗🧍🏽‍♂️🚌")
st.markdown("Choose input mode to detect objects using YOLOv8:")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Lightweight model for speed

model = load_model()

mode = st.radio("Select Input Mode", ["Upload Image/Video", "Use Webcam"])


if mode == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4'])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if 'image' in file_type:
            file_bytes = uploaded_file.read()
            np_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            results = model(np_img)
            annotated = results[0].plot()
            st.image(annotated, caption="Detected Image", use_container_width=True)

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
                stframe.image(annotated_frame, channels="BGR")
            cap.release()

elif mode == "Use Webcam":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annotated = results[0].plot()
            return annotated

    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)
    

