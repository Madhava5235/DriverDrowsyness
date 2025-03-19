import streamlit as st
import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model("drowsiness_cnn_model.h5")

# Initialize pygame for alarm
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

def detect_drowsiness():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not access the webcam.")
            break

        # Convert to grayscale and preprocess
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (64, 64))
        gray_frame = gray_frame.reshape(1, 64, 64, 1) / 255.0  # Normalize

        # Predict using CNN model
        prediction = model.predict(gray_frame)[0][0]

        # Display Prediction value
        st.write(f"Prediction: {prediction:.4f}")

        # If prediction > 0.7 (drowsy), play alarm
        if prediction > 0.7:
            cv2.putText(frame, "DROWSINESS DETECTED!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()
            cv2.putText(frame, "Awake", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)
        
        # Wait briefly to prevent UI lag
        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

# Streamlit UI
st.title("Driver Drowsiness Detection")
st.write("Click the button below to start real-time detection.")

if st.button("Start Detection"):
    detect_drowsiness()
