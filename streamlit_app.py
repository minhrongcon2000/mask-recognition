import time
import cv2
import dlib
import numpy as np
import os
import gdown

from utils import detect_face, extract_feature
from tensorflow.keras.models import load_model
import streamlit as st

import config

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(config.prototxtPath, config.weightsPath)

# nose model
print("[INFO] - loading nose model...")
if not os.path.exists(config.noseModelPath):
    gdown.download(
        "https://drive.google.com/file/d/15degH8iQviKHVY3k1Sq5ePxrmxKViGBZ/view?usp=sharing", fuzzy=True)

noseModel = load_model(config.noseModelPath)


# load face feature extractor, use for nose detection
feature_extractor = dlib.shape_predictor(config.featureExtractorPath)

num_frames = 0  # counter for how many frames have passed
save_period = 50  # period of recognizing face and storing data


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

# nose model
print("[INFO] - loading nose model...")
noseModel = load_model(config.noseModelPath)

st.title("Mask face app")


# load face feature extractor, use for nose detection
feature_extractor = dlib.shape_predictor(config.featureExtractorPath)

num_frames = 0  # counter for how many frames have passed
save_period = 50  # period of recognizing face and storing data

st.write("This is an application for showing correct mask usage!")
run = st.checkbox("Run the task")
frame_window = st.image([])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while run:
    ret, frame = vs.read()

    num_frames = (num_frames + 1) % save_period

    locs = detect_face(frame, faceNet)
    noses = []
    nose_locs = []

    for box in locs:
        (startX, startY, endX, endY) = box
        color = (0, 255, 0)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_features = feature_extractor(
            gray_frame, dlib.rectangle(startX, startY, endX, endY))

        # backupFrame = frame.copy()

        # nose part
        noseStartX, noseStartY, noseEndX, noseEndY = extract_feature(
            face_features, (27, 35))
        nose = frame[noseStartY:noseEndY, noseStartX:noseEndX]
        nose = cv2.resize(nose, (224, 224))
        nose = nose / 255.
        noses.append(nose)
        nose_locs.append((noseStartX, noseStartY, noseEndX, noseEndY))

    noses = np.array(noses)
    if noses.any():
        nosePredictions = noseModel.predict(noses)
        for i, (noseStartX, noseStartY, noseEndX, noseEndY) in enumerate(nose_locs):
            covered, uncovered = nosePredictions[i]
            label = "Nose: {}".format(
                "uncovered" if covered <= uncovered else "covered")
            maskLabel = "Mask: incorrect usage" if covered <= uncovered else "Mask: correct usage"
            cv2.putText(frame, label, (noseStartX - 10, noseEndY + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, maskLabel, (locs[i][0], locs[i][1]),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (noseStartX, noseStartY),
                          (noseEndX, noseEndY), (0, 255, 0), 2)

    frame_window.image(frame)

else:
    st.write("Stopped")
