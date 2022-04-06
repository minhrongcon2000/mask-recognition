import argparse
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

import config
import dlib
from tensorflow.keras.models import load_model
from utils import detect_face, extract_feature

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(config.prototxtPath, config.weightsPath)

# nose model
print("[INFO] - loading nose model...")
noseModel = load_model(config.noseModelPath)


# load face feature extractor, use for nose detection
feature_extractor = dlib.shape_predictor(config.featureExtractorPath)

IMG_PATH = "dataset/CMFD/00011_Mask.jpg"
img = cv2.imread(IMG_PATH)

locs = detect_face(img, faceNet, return_confident=True)

chosen_loc = max(locs, key=lambda item: item[1])
(startX, startY, endX, endY), _ = chosen_loc

nose_locs = []
noses = []

color = (0, 255, 0)
gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_features = feature_extractor(
    gray_frame, dlib.rectangle(startX, startY, endX, endY))

# nose part
noseStartX, noseStartY, noseEndX, noseEndY = extract_feature(
    face_features, (27, 35))
nose = img[noseStartY:noseEndY, noseStartX:noseEndX]
nose = cv2.resize(nose, (224, 224))
nose = nose / 255.
noses.append(nose)
nose_locs.append((noseStartX, noseStartY, noseEndX, noseEndY))
cv2.rectangle(img, (startX, startY), (endX, endY), color, 4)

noses = np.array(noses)
if noses.any():
    nosePredictions = noseModel.predict(noses)
    for i, (noseStartX, noseStartY, noseEndX, noseEndY) in enumerate(nose_locs):
        covered, uncovered = nosePredictions[i]
        label = "Nose: {}".format(
            "uncovered" if abs(uncovered - covered) <= 0.2 else "covered")
        cv2.putText(img, label, (noseStartX - 50, noseStartY - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (noseStartX, noseStartY),
                      (noseEndX, noseEndY), (0, 255, 0), 4)

cv2.imwrite("demo.jpg", img)
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
