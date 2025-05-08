# Real-Time-Object-Detection-with-YOLOv8

This is a real-time object detection app powered by **YOLOv8**, built with **Streamlit**. The app allows users to upload images or videos or use their webcam to detect multiple objects, such as vehicles, pedestrians, and other items, in real time. It's designed for both efficiency and ease of use.

## Features

- **Object Detection on Images/Videos**: Upload images or videos to detect objects.
- **Webcam Detection**: Use your webcam to detect objects in real-time.
- **Fast and Lightweight**: Powered by YOLOv8n (the lightweight version of YOLOv8) for fast inference.
- **Interactive Interface**: Built using Streamlit for easy access and interaction.

## Technologies Used

- **YOLOv8**: A state-of-the-art object detection model known for its accuracy and speed.
- **Streamlit**: A Python framework to create web applications for machine learning models.
- **OpenCV**: Used for image and video processing.
- **NumPy**: A core library for handling arrays and images.

## Setup Instructions

To run this project locally, follow these steps:

### 1. Clone the Repository
Clone this repository to your local machine:
- bash
git clone https://github.com/yourusername/YOLOv8-Object-Detection-App.git
cd Real-Time-Object-Detection-with-YOLOv8 

### 2 Install Dependencies
pip install -r requirements.txt


### 3 Run the App
streamlit run app.py


### 4. Upload an Image/Video or Use Webcam

- **Choose "Upload Image/Video"** to upload your media file.
- **Choose "Use Webcam"** to detect objects in real-time using your webcam.

## Notes

- **Performance**: Use low-resolution images or short videos for better performance.
- **YOLOv8n**: This app uses the YOLOv8n model, the lightweight version, for fast inference.
- **Webcam Detection**: Ensure your webcam is connected and accessible for real-time object detection.


## Acknowledgements

- **YOLOv8**: Thanks to Ultralytics for the YOLOv8 model.
- **Streamlit**: For providing an amazing framework for creating web applications.


