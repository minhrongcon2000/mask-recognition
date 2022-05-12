from imutils.video import VideoStream
import imutils
import time
import cv2
import os
import dlib
import numpy as np

from utils import detect_face
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from datetime import datetime

import config

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(config.prototxtPath, config.weightsPath)

# nose model
print("[INFO] - loading nose model...")
faceModel = load_model(config.faceModelPath)

# num_frames = 0  # counter for how many frames have passed
# save_period = 50  # period of recognizing face and storing data


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

captureTime = []

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # num_frames = (num_frames + 1) % save_period

    locs = detect_face(frame, faceNet)
    
    faces = []

    for box in locs:
        (startX, startY, endX, endY) = box
        color = (0, 255, 0)
        cv2.rectangle(frame, (startX, startY),
                      (endX, endY), (0, 255, 0), 2)
        face = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        faces.append(face)

    faces = np.array(faces)

    if faces.any():
        # start = time.time()
        facePred = faceModel.predict(faces)
        # end = time.time()
        # captureTime.append(end - start)
        for i, (startX, startY, endX, endY) in enumerate(locs):
            covered, uncovered = facePred[i]
            label = "incorrect" if uncovered > covered else "correct"
            percentage = uncovered if uncovered > covered else covered
            cv2.putText(frame, f"Mask: {label} ({percentage * 100:.2f}%)", (startX - 40, startY - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0) if label == "correct" else (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY),
                        (endX, endY), (0, 255, 0) if label == "correct" else (0, 0, 255), 2)

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
# print("Number samples: {}".format(len(captureTime)))
# print("Mean time: {}".format(np.mean(captureTime)))
# print("Std: {}".format(np.std(captureTime, ddof=1)))
