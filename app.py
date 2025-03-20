import streamlit as st
import numpy as np
import av
import cv2
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# Load model only once (cached)
@st.cache_resource
def load_drowsiness_model():
    return load_model("drowsiness_cnn_model.h5")

model = load_drowsiness_model()

# Load Haarcascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# WebRTC Configuration for Low Latency
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Video Processing Class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_skip = 3  # Process every 3rd frame (optimizing latency)
        self.counter = 0

    def recv(self, frame):
        self.counter += 1
        if self.counter % self.frame_skip != 0:
            return frame  # Skip processing some frames for performance

        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Reduce resolution for performance
        img_resized = cv2.resize(img, (640, 480))  
        gray_resized = cv2.resize(gray, (640, 480))

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_resized, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(img_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Region of interest for eyes
            roi_gray = gray_resized[y:y + h, x:x + w]
            roi_color = img_resized[y:y + h, x:x + w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Preprocess for model input
        gray_frame = cv2.resize(gray_resized, (64, 64)).reshape(1, 64, 64, 1) / 255.0  
        prediction = model.predict(gray_frame, verbose=0)[0][0]  

        # Overlay prediction
        label = "DROWSY" if prediction > 0.7 else "AWAKE"
        color = (0, 0, 255) if prediction > 0.7 else (0, 255, 0)
        cv2.putText(img_resized, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img_resized, format="bgr24")

# Streamlit UI
st.title("🚗 Driver Drowsiness Detection")
st.write("Click 'Start' to begin real-time detection.")

webrtc_streamer(
    key="drowsiness_detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 20}, "audio": False}  
)
