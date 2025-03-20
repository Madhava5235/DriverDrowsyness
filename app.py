import streamlit as st
import numpy as np
import av
import cv2
import dlib  # For face and eye landmark detection
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Load model only once (cached)
@st.cache_resource
def load_drowsiness_model():
    return load_model("drowsiness_cnn_model.h5")

model = load_drowsiness_model()

# Load Dlib's pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is available

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    """Calculate the Eye Aspect Ratio (EAR) to detect eye closure."""
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

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

        # Convert image to grayscale
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray_frame)
        if len(faces) == 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")  # No face detected

        for face in faces:
            landmarks = predictor(gray_frame, face)

            # Extract eye landmarks
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
            right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

            # Compute EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Preprocess image for CNN model
            resized_frame = cv2.resize(gray_frame, (64, 64))
            normalized_frame = resized_frame.reshape(1, 64, 64, 1) / 255.0  # Normalize

            # Predict drowsiness
            cnn_prediction = model.predict(normalized_frame)[0][0]

            # Adaptive thresholding based on CNN & EAR
            if cnn_prediction > 0.7 or avg_ear < 0.22:  # 0.22 EAR is a common threshold for drowsiness
                self.drowsy_frame_count += 1
            else:
                self.drowsy_frame_count = 0

            # Overlay prediction
            label = "DROWSY" if self.drowsy_frame_count >= self.alert_threshold else "AWAKE"
            color = (0, 0, 255) if label == "DROWSY" else (0, 255, 0)
            cv2.putText(img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Draw face & eye landmarks
            for point in left_eye:
                cv2.circle(img, tuple(point), 2, (0, 255, 0), -1)
            for point in right_eye:
                cv2.circle(img, tuple(point), 2, (0, 255, 0), -1)
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

            # Streamlit alert if drowsy for consecutive frames
            if self.drowsy_frame_count >= self.alert_threshold:
                st.warning("⚠️ Drowsiness Detected! Stay Alert!")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("🚗 Enhanced Driver Drowsiness Detection")
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
