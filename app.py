import os
import tensorflow as tf

# Fix OpenCV `libGL.so.1` error by setting headless mode
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# Suppress TensorFlow warnings related to feedback tensors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = All messages, 1 = INFO, 2 = WARNINGS, 3 = ERRORS
tf.get_logger().setLevel('ERROR')

import streamlit as st
import numpy as np
import av
import cv2
import mediapipe as mp  # Using MediaPipe for face & eye detection
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Load MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load model only once (cached)
@st.cache_resource
def load_drowsiness_model():
    return load_model("drowsiness_cnn_model.h5")

model = load_drowsiness_model()

# Function to extract eye landmarks from MediaPipe FaceMesh
def get_eye_landmarks(image):
    """Detects eyes using MediaPipe FaceMesh instead of dlib"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = [face_landmarks.landmark[i] for i in range(133, 144)]
            right_eye = [face_landmarks.landmark[i] for i in range(362, 373)]
            return left_eye, right_eye
    return None, None

# Video Processing Class for WebRTC
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_skip = 2  # Process every 2nd frame for better performance
        self.counter = 0
        self.drowsy_frame_count = 0
        self.alert_threshold = 5  # Number of consecutive drowsy frames before alert

    def recv(self, frame):
        self.counter += 1
        if self.counter % self.frame_skip != 0:
            return frame  # Skip processing some frames

        img = frame.to_ndarray(format="bgr24")

        # Detect eye landmarks
        left_eye, right_eye = get_eye_landmarks(img)
        if left_eye is None or right_eye is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")  # No face detected

        # Convert to grayscale and preprocess for CNN model
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (64, 64))
        normalized_frame = resized_frame.reshape(1, 64, 64, 1) / 255.0  # Normalize

        # Predict drowsiness using CNN model
        cnn_prediction = model.predict(normalized_frame)[0][0]

        # Adaptive thresholding based on CNN prediction
        if cnn_prediction > 0.7:
            self.drowsy_frame_count += 1
        else:
            self.drowsy_frame_count = 0

        # Overlay prediction
        label = "DROWSY" if self.drowsy_frame_count >= self.alert_threshold else "AWAKE"
        color = (0, 0, 255) if label == "DROWSY" else (0, 255, 0)
        cv2.putText(img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw eye landmarks
        for point in left_eye:
            x, y = int(point.x * img.shape[1]), int(point.y * img.shape[0])
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        for point in right_eye:
            x, y = int(point.x * img.shape[1]), int(point.y * img.shape[0])
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # Streamlit alert if drowsy for consecutive frames
        if self.drowsy_frame_count >= self.alert_threshold:
            st.warning("⚠️ Drowsiness Detected! Stay Alert!")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("🚗 Enhanced Driver Drowsiness Detection with MediaPipe")
st.write("Click 'Start' to begin real-time detection.")

webrtc_streamer(
    key="drowsiness",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},  # Prevents session conflicts
    frontend_rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        "iceTransportPolicy": "relay"  # Forces TCP transport to avoid `sendto` errors
    },
    video_html_attrs={"autoPlay": True, "controls": False, "muted": True}  # Prevents browser autoplay issues
)
