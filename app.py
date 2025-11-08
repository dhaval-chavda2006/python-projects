import streamlit as st
import cv2
import numpy as np
from PIL import Image

# PAGE CONFIG
st.set_page_config(page_title="Image Recognition App", layout="centered")


custom_css = """
<style>
body {
    background-color: #000000;
    color: #FFD700;
}
.stApp {
    background-color: #000000;
}
h1, h2, h3, label {
    color: #FFD700 !important;
}
.uploadedFile {
    color: #FFD700 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# TITLE
st.title("Image Recognition App {Object Detection}")
st.write("Upload an image and the system will detect and label objects in it.")

# MODEL LOAD
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn.DetectionModel(frozen_model, config_file)

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Load Labels
with open("Labels.txt", "rt") as f:
    classLabels = f.read().rstrip("\n").split("\n")

# FILE UPLOADER 
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detect Objects
    ClassIndex, confidence, bbox = model.detect(img_cv, confThreshold=0.5)

    if len(ClassIndex) != 0:
        for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            cv2.rectangle(img, box, (255, 215, 0), 2)  # Golden Border
            cv2.putText(img, classLabels[ClassInd - 1], (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)

    # Convert back to RGB for display
    final_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(final_img, caption="Detected Image", use_column_width=True)
