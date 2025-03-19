import os
import streamlit as st
import numpy as np
import av
import cv2
import time
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Get the absolute path of the working directory
BASE_DIR = os.getcwd()

# Define file paths
MODEL_PATH = os.path.join(BASE_DIR, "drowsiness_cnn_model.h5")
ALARM_PATH = os.path.join(BASE_DIR, "alarm.mp3")

# Load model safely
@st.cache_resource
def load_drowsiness_model():
    if not os.path.exists(MODEL_PATH):
        st.error("🔴 Error: Model file not found! Ensure 'drowsiness_cnn_model.h5' exists in your GitHub repo.")
        return None
    return load_model(MODEL_PATH)

model = load_drowsiness_model()

# Load alarm sound safely
if os.path.exists(ALARM_PATH):
    with open(ALARM_PATH, "rb") as f:
        alarm_sound = f.read()
else:
    st.warning("⚠️ Warning: 'alarm.mp3' not found! Alarm sound will not play.")

# Video Processing Class for WebRTC
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_skip = 3  # Process every 3rd frame for better performance
        self.counter = 0
        self.last_alarm_time = 0  # Prevent continuous alarm spam

    def recv(self, frame):
        self.counter += 1
        if self.counter % self.frame_skip != 0:
            return frame  # Skip processing some frames

        img = frame.to_ndarray(format="bgr24")

        # Preprocess image
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (64, 64))
        gray_frame = gray_frame.reshape(1, 64, 64, 1) / 255.0  # Normalize

        # Predict drowsiness
        if model is not None:
            prediction = model.predict(gray_frame)[0][0]
        else:
            prediction = 0  # Default to 'AWAKE' if the model is missing

        # Overlay prediction
        label = "DROWSY" if prediction > 0.7 else "AWAKE"
        color = (0, 0, 255) if prediction > 0.7 else (0, 255, 0)
        cv2.putText(img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Play alarm if drowsy (with cooldown)
        if prediction > 0.7 and (time.time() - self.last_alarm_time > 3):  # 3-second cooldown
            self.last_alarm_time = time.time()
            if os.path.exists(ALARM_PATH):
                st.audio(alarm_sound, format="audio/mp3", start_time=0)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("🚗 Driver Drowsiness Detection")
st.write("Click 'Start' to begin real-time detection.")

webrtc_streamer(key="drowsiness", video_processor_factory=VideoProcessor)
