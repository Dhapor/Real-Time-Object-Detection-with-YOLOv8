import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# Setting up the page configuration
st.set_page_config(page_title="YOLOv8 Object Detection App")
st.title("YOLOv8 Object Detection App 🚗🧍🏽‍♂️🚌")
st.markdown("Choose input mode to detect objects using YOLOv8:")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

mode = st.radio("Select Input Mode", ["Upload Image/Video", "Use Webcam"])

if mode == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4', 'mov'])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        # For Image files
        if 'image' in file_type:
            file_bytes = uploaded_file.read()
            np_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            results = model(np_img)
            annotated = results[0].plot()
            st.image(annotated, caption="Detected Image", use_container_width=True)

        # For Video files
        elif 'video' in file_type:
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mov" if 'quicktime' in file_type else ".mp4")
            temp_input.write(uploaded_file.read())
            temp_input.close()

            # Convert MOV to MP4 if necessary
            if 'quicktime' in file_type:
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                clip = VideoFileClip(temp_input.name)
                clip.write_videofile(temp_output.name, codec='libx264')
                video_path = temp_output.name
            else:
                video_path = temp_input.name

            # Open the video file
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            # Read and process video frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Get the frame's original aspect ratio
                height, width, _ = frame.shape
                aspect_ratio = width / height

                # Resize the frame to fit the container, maintaining the original aspect ratio
                new_width = 600  # You can set a fixed width
                new_height = int(new_width / aspect_ratio)  # Calculate corresponding height
                resized_frame = cv2.resize(frame, (new_width, new_height))

                # Run object detection
                results = model(resized_frame)
                annotated_frame = results[0].plot()

                # Display the annotated frame
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()

elif mode == "Use Webcam":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annotated = results[0].plot()
            return VideoFrame.from_ndarray(annotated_img, format="bgr24")

    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)
