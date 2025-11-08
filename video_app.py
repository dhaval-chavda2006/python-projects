import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Video Object Detection", layout="wide")
st.title("Object Detection For Video")

# Load Model
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn.DetectionModel(frozen_model, config_file)

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Load Labels
with open("Labels.txt", "rt") as f:
    classLabels = f.read().strip().split("\n")

# UI
option = st.radio("Choose Video Source:", ["Webcam", "Upload Video"])
frame_placeholder = st.empty()

if option == "Webcam":
    cap = cv2.VideoCapture(0)

elif option == "Upload Video":
    file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if file:
        tfile = open("uploaded_video.mp4", "wb")
        tfile.write(file.read())
        cap = cv2.VideoCapture("uploaded_video.mp4")
    else:
        st.stop()

# Process Video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)

    if len(ClassIndex) != 0:
        for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, box, (0, 255, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

cap.release()
