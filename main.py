import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time

# Load the YOLOv8 model from the local path
model_path = 'E:/Data science project/License_plate_detection_yolov8/best.pt'
model = YOLO(model_path)

# Define the function for detection
def detect_license_plate(image):
    img = np.array(image)
    results = model(img)

    # Draw bounding boxes on the image
    for result in results[0].boxes:
        if result.cls == 0:  # Assuming 0 is the class index for LicensePlate
            xmin, ymin, xmax, ymax = result.xyxy[0]  # Get bounding box coordinates
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    img = Image.fromarray(img)
    return img

# Streamlit UI
st.title("License Plate Detection")

st.sidebar.header("Upload Image or Video")
uploaded_file = st.sidebar.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

# Button to start processing
process_button = st.sidebar.button("Process")

if process_button:
    if uploaded_file is not None:
        if uploaded_file.type in ["jpg", "jpeg", "png"]:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                processed_image = detect_license_plate(image)
                st.image(processed_image, caption='Processed Image with Detection', use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {e}")

        elif uploaded_file.type == "mp4":
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)

                # Get total number of frames
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                stframe = st.empty()
                start_time = time.time()
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert frame to PIL Image for model processing
                    frame_pil = Image.fromarray(frame)
                    processed_frame_pil = detect_license_plate(frame_pil)

                    # Convert back to OpenCV format for Streamlit display
                    processed_frame = np.array(processed_frame_pil)
                    stframe.image(processed_frame, channels="BGR", use_column_width=True)

                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    estimated_time_remaining = (elapsed_time / frame_count) * (total_frames - frame_count)

                    # Display estimated time remaining
                    st.sidebar.write(f"Estimated Time Remaining: {int(estimated_time_remaining)} seconds")

                cap.release()
                os.remove(tfile.name)
            except Exception as e:
                st.error(f"Error processing video: {e}")

# Button to start camera
if st.sidebar.button("Start Camera"):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image for model processing
        frame_pil = Image.fromarray(frame)
        processed_frame_pil = detect_license_plate(frame_pil)

        # Convert back to OpenCV format for Streamlit display
        processed_frame = np.array(processed_frame_pil)
        stframe.image(processed_frame, channels="BGR", use_column_width=True)

    cap.release()
