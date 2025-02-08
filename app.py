import cv2
import numpy as np
import streamlit as st
import tempfile
import time

# Convert the color of the image
def convert_color(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Set the title and sidebar
st.set_page_config(page_title="Object Tracking Application", page_icon="🎥", layout="wide")
st.title("🎥 Object Tracking Application")
st.sidebar.title("Upload and Settings")

# File uploader
upload = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Settings
min_contour_area = st.sidebar.slider("Minimum Contour Area", 100, 1000, 500)
box_color = st.sidebar.color_picker("Bounding Box Color", "#00FF00")
frame_rate = st.sidebar.slider("Frame Rate (ms)", 1, 100, 3)
display_option = st.sidebar.radio("Display Options", ("Both Video and Foreground Mask", "Only Video with Object Tracking", "Only Foreground Mask"))

if upload is not None:
    # Save the video into a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload.read())
    tfile.close()
    
    captures = cv2.VideoCapture(tfile.name)
    if not captures.isOpened():
        st.error("Error: Cannot open file")
    else:
        stframe = st.empty()  # Placeholder for the video frame
        stmask = st.empty()  # Placeholder for the foreground mask
        back_sub = cv2.createBackgroundSubtractorMOG2()
        while captures.isOpened():
            ret, frame = captures.read()
            if not ret:
                break
            fg_mask = back_sub.apply(frame)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in contours:
                if cv2.contourArea(i) > min_contour_area:
                    x, y, width, height = cv2.boundingRect(i)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), tuple(int(box_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), 2)
            
            if display_option == "Both Video and Foreground Mask":
                stframe.image(convert_color(frame), caption="Video with Object Tracking")
                stmask.image(fg_mask, caption="Foreground Mask", channels="GRAY")
            elif display_option == "Only Video with Object Tracking":
                stframe.image(convert_color(frame), caption="Video with Object Tracking")
            elif display_option == "Only Foreground Mask":
                stmask.image(fg_mask, caption="Foreground Mask", channels="GRAY")
            
            time.sleep(frame_rate / 1000)
        captures.release()
else:
    st.warning("Please upload a video to start object tracking.")