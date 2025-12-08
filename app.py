import streamlit as st
import cv2
import numpy as np
from av import VideoFrame
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    layout="wide"
)

st.title("🚦 YOLOv8 Object Detection App")
st.markdown("Upload media or use your webcam to detect objects using YOLOv8")

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------------------------------
# INPUT OPTIONS
# -----------------------------------------------------
mode = st.radio("Select Mode 👇", ["Upload Image/Video", "Use Webcam"])


# =====================================================
# UPLOAD MODE
# =====================================================
if mode == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload an image or video", 
                                     type=['jpg', 'jpeg', 'png', 'mp4', 'mov'])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        # IMAGE
        if "image" in file_type:
            file_bytes = uploaded_file.read()
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

            results = model(img)
            annotated = results[0].plot()
            st.image(annotated, caption="Detected Objects", use_container_width=True)

        # VIDEO
        elif "video" in file_type:
            import tempfile
            from moviepy.editor import VideoFileClip

            temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mov" if "quicktime" in file_type else ".mp4")
            temp_in.write(uploaded_file.read())
            temp_in.close()

            # Convert MOV -> MP4
            if "quicktime" in file_type:
                temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                clip = VideoFileClip(temp_in.name)
                clip.write_videofile(temp_out.name, codec='libx264')
                video_path = temp_out.name
            else:
                video_path = temp_in.name

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))
                results = model(frame)
                annotated = results[0].plot()

                stframe.image(annotated, channels="RGB", use_container_width=True)

            cap.release()


# =====================================================
# LIVE WEBCAM MODE
# =====================================================
elif mode == "Use Webcam":

    class YOLOTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annot = results[0].plot()
            return VideoFrame.from_ndarray(annot, format="bgr24")

    webrtc_streamer(
        key="yolo-live",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=YOLOTransformer,
        media_stream_constraints={"video": True, "audio": False},

        # -----------------------------------------------------
        # 🔥 TURN + STUN CONFIG (STREAMLIT CLOUD WORKING)
        # -----------------------------------------------------
        rtc_configuration={
            "iceServers": [
                # STUN
                {"urls": ["stun:stun.l.google.com:19302"]},

                # TURN (REQUIRED FOR STREAMLIT CLOUD)
                {
                    "urls": ["turn:relay1.expressturn.com:3478?transport=tcp"],
                    "username": "efRbyMHX2e1UjICgcr0M0Q",
                    "credential": "3A4zubiLWMp0Y3C47XPEWQ"
                }
            ],
            "iceTransportPolicy": "relay"
        },
        async_processing=True,
    )
