from imutils.video import VideoStream
import imutils
import time
import cv2
import os
import dlib
import numpy as np

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

num_frames = 0  # counter for how many frames have passed
save_period = 50  # period of recognizing face and storing data


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    num_frames = (num_frames + 1) % save_period

    locs = detect_face(frame, faceNet)
    noses = []
    nose_locs = []

    for box in locs:
        (startX, startY, endX, endY) = box
        color = (0, 255, 0)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_features = feature_extractor(
            gray_frame, dlib.rectangle(startX, startY, endX, endY))

        backupFrame = frame.copy()

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
                "uncovered" if abs(uncovered - covered) <= 0.2 else "covered")
            print(covered, uncovered, label)
            cv2.putText(frame, label, (noseStartX - 10, noseStartY),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (noseStartX, noseStartY),
                          (noseEndX, noseEndY), (0, 255, 0), 2)

            # if num_frames == 0:
            #     if covered > uncovered:
            #         cv2.imwrite(os.path.join(
            #             config.coveredPath, "{}.png".format(datetime.now())), backupFrame)
            #     else:
            #         cv2.imwrite(os.path.join(
            #             config.uncoveredPath, "{}.png".format(datetime.now())), backupFrame)
            #         print("Send to guard!")

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
