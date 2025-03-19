import streamlit as st
import numpy as np
import av
import cv2
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Load the trained model
MODEL_PATH = "drowsiness_cnn_model.h5"
model = load_model(MODEL_PATH)

# Load alarm sound
with open("alarm.mp3", "rb") as f:
    alarm_sound = f.read()

# Video Processing Class for WebRTC
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess image
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (64, 64))
        gray_frame = gray_frame.reshape(1, 64, 64, 1)
        gray_frame = gray_frame / 255.0  # Normalize

        # Predict drowsiness
        prediction = model.predict(gray_frame)[0][0]

        # Overlay prediction
        label = "DROWSY" if prediction > 0.7 else "AWAKE"
        color = (0, 0, 255) if prediction > 0.7 else (0, 255, 0)
        cv2.putText(img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Play alarm if drowsy
        if prediction > 0.7:
            st.audio(alarm_sound, format="audio/mp3", start_time=0)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("🚗 Driver Drowsiness Detection")
st.write("Click 'Start' to begin real-time detection.")

webrtc_streamer(key="drowsiness", video_processor_factory=VideoProcessor)
