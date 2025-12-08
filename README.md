# Real-Time Object Detection with YOLOv8

A simple object detection app using **YOLOv8** and **Streamlit**. Detect objects in images, MP4 videos, or your local webcam (webcam works only locally).

## Features

- Detect objects in images and MP4 videos
- Use your webcam for real-time detection (local only)
- Annotated results can be downloaded
- Frames preserve aspect ratio and fit on screen

## Requirements

- Python 3.10+
- streamlit, ultralytics, opencv-python-headless, numpy, av

**requirements.txt:**
streamlit==1.27.0
ultralytics==8.1.16
opencv-python-headless==4.8.1.78
numpy==1.26.0
av==11.1.0

## How to Use

Clone the repo:
git clone https://github.com/yourusername/YOLOv8-Object-Detection-App.git
cd YOLOv8-Object-Detection-App

Install dependencies:
pip install -r requirements.txt

Run the app locally:
streamlit run app.py

Use the app:
Select Upload Image/Video to process files
Select Use Webcam for real-time detection (local only)
Download annotated images or videos when done

Note: Webcam does not work on Streamlit Cloud. Use only image/video uploads there.
