import cv2
import numpy as np
import streamlit as st
import tempfile
import time


captures = cv2.VideoCapture(r"D:\Work\Route\sesssss\Object_tracking_project2\vtest.avi")
back_sub = cv2.createBackgroundSubtractorMOG2()

while captures.isOpened():
    ret, frame = captures.read()
    if not ret:
        break
    fg_mask = back_sub.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        if cv2.contourArea(i) > 500:
            x, y, width, height = cv2.boundingRect(i)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (15, 0, 255), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fg_mask)
    if cv2.waitKey(30) & 0xFF == 27:
        break
    
captures.release()
cv2.destroyAllWindows()



def convert_color(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


st.title("Object Tracking Application")

upload = st.file_uploader("Upload an video", type=["mp4", "avi", "mov"])

if upload is not None:
    #save the video into a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload.read())
    tfile.close()
    
    captures = cv2.VideoCapture(tfile.name)
    if not captures.isOpened():
        st.write("Error: Cannot open file")
        
    else:
        stframe = st.empty() #placeholder for the video frame
        back_sub = cv2.createBackgroundSubtractorMOG2()
        while captures.isOpened():
            ret, frame = captures.read()
            if not ret:
                break
            fg_mask = back_sub.apply(frame)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in contours:
                if cv2.contourArea(i) > 500: 
                    x, y, width, height = cv2.boundingRect(i)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            stframe.image(convert_color(frame))
            time.sleep(0.003)
        captures.release()
else:
    st.warning("please: add a video")