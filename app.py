import streamlit as st
import cv2
from av import VideoFrame
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# Page config
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

        # Handle images
        if 'image' in file_type:
            file_bytes = uploaded_file.read()
            np_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            results = model(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))  # Convert to RGB before inference
            annotated = results[0].plot()
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            st.image(annotated_bgr, caption="Detected Image", channels="BGR", use_container_width=True)

        # Handle videos
        elif 'video' in file_type:
            import tempfile
            from moviepy.editor import VideoFileClip

            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mov" if 'quicktime' in file_type else ".mp4")
            temp_input.write(uploaded_file.read())
            temp_input.close()

            # Convert MOV to MP4 if needed
            if 'quicktime' in file_type:
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                clip = VideoFileClip(temp_input.name)
                clip.write_videofile(temp_output.name, codec='libx264')
                video_path = temp_output.name
            else:
                video_path = temp_input.name

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                height, width, _ = frame.shape
                aspect_ratio = width / height

                new_width = 600
                new_height = int(new_width / aspect_ratio)
                resized_frame = cv2.resize(frame, (new_width, new_height))

                results = model(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                annotated = results[0].plot()
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

                stframe.image(annotated_bgr, channels="BGR", use_container_width=True)

            cap.release()

elif mode == "Use Webcam":

    class YOLOTransformer(VideoTransformerBase):
        def transform(self, frame):
            try:
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = model(img_rgb)
                annotated_rgb = results[0].plot()
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                return VideoFrame.from_ndarray(annotated_bgr, format="bgr24")
            except Exception as e:
                print(f"Error in transform: {e}")
                return frame.to_ndarray(format="bgr24")

    webrtc_streamer(
        key="yolo-live",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=YOLOTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
