import streamlit as st
import cv2
import torch
import tempfile
import os
from PIL import Image

# Streamlit UI
st.title("Object Detection with YOLOv5")
st.sidebar.title("Upload Video")

uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Load YOLOv5 model (use your own yolov5l.pt model path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m.pt')  # Load your model

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_window = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection with YOLOv5
        results = model(frame)  # Get detection results from YOLOv5

        # Render results on the frame (bounding boxes, labels, etc.)
        frame = results.render()[0]

        # Display frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

if uploaded_video is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_video.read())
        temp_video_path = temp_file.name

    st.sidebar.success("Video uploaded successfully!")
    st.video(temp_video_path)

    if st.button("Start Detection"):
        process_video(temp_video_path)
        os.remove(temp_video_path)
else:
    st.sidebar.info("Please upload a video file to proceed.")
