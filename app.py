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

# JavaScript for buzzer with speaker selection
buzzer_html = """
<audio id="buzzer" src="https://www.soundjay.com/button/beep-07.wav"></audio>
<script>
function playBuzzer() {
    let audio = document.getElementById("buzzer");
    audio.play();
}
</script>
"""

# Video Processing Class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_skip = 2  # Process every 2nd frame for better performance
        self.counter = 0

    def recv(self, frame):
        self.counter += 1
        if self.counter % self.frame_skip != 0:
            return frame  # Skip processing some frames for performance

        img = frame.to_ndarray(format="bgr24")  # Maintain original resolution
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Region of interest for eyes
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Preprocess for model input
        resized_gray = cv2.resize(gray, (64, 64)).reshape(1, 64, 64, 1) / 255.0  
        prediction = model.predict(resized_gray, verbose=0)[0][0]  

        # Overlay prediction
        label = "DROWSY" if prediction > 0.7 else "AWAKE"
        color = (0, 0, 255) if prediction > 0.7 else (0, 255, 0)
        cv2.putText(img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # If drowsiness detected, set flag for buzzer
        if prediction > 0.7:
            st.session_state["play_buzzer"] = True

        return av.VideoFrame.from_ndarray(img, format="bgr24")  # Preserve original resolution

# Streamlit UI
st.title("🚗 Driver Drowsiness Detection")
st.write("Click 'Start' to begin real-time detection. If drowsiness is detected, a buzzer will sound.")

# Camera selection dropdown
camera_options = {
    "Default Camera": {},
    "Front Camera (Mobile)": {"video": {"facingMode": "user"}},
    "Rear Camera (Mobile)": {"video": {"facingMode": "environment"}}
}
selected_camera = st.selectbox("📷 Select Camera", list(camera_options.keys()))

webrtc_streamer(
    key="drowsiness_detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints=camera_options[selected_camera]  # Apply selected camera
)

# Inject JavaScript for buzzer
st.components.v1.html(buzzer_html, height=0)

# Speaker selection (Manual trigger to avoid autoplay restrictions)
st.markdown("🔊 **Select Speaker & Play Buzzer**")
if st.button("🔔 Play Test Buzzer"):
    st.markdown("<script>playBuzzer();</script>", unsafe_allow_html=True)

# Play buzzer if drowsiness detected
if st.session_state.get("play_buzzer", False):
    st.markdown("<script>playBuzzer();</script>", unsafe_allow_html=True)
    st.session_state["play_buzzer"] = False  # Reset after playing sound
