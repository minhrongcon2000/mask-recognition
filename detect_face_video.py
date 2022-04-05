from imutils.video import VideoStream
import imutils
import time
import cv2
import os
import dlib
import numpy as np
from mask_recognition import make_inference_from_frame

from utils import detect_face, extract_feature
from tensorflow.keras.models import load_model
from datetime import datetime

import config

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(config.prototxtPath, config.weightsPath)

# nose model
print("[INFO] - loading nose model...")
noseModel = load_model(config.noseModelPath)


# load face feature extractor, use for nose detection
feature_extractor = dlib.shape_predictor(config.featureExtractorPath)


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    results = make_inference_from_frame(frame)

    if results is not None:
        face_locs, nose_locs, nose_preds = results

        for (startX, startY, endX, endY) in face_locs:
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

        for nose_loc, nose_pred in zip(nose_locs, nose_preds):
            startX, startY, endX, endY = nose_loc
            covered, uncovered = nose_pred
            label = "Nose: {}".format("uncovered" if abs(
                covered - uncovered) <= 0.2 else "covered")
            print(covered, uncovered, label)
            cv2.putText(frame, label, (startX - 10, startY),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
