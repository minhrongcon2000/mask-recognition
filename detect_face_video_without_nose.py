from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np
import mediapipe as mp

from utils import detect_face
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
import math

import config

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()

# nose model
print("[INFO] - loading nose model...")
faceModel = load_model(config.faceModelPath)

# num_frames = 0  # counter for how many frames have passed
# save_period = 50  # period of recognizing face and storing data


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

avgFPS = 0
frameCount = 0

while True:
    frameCount += 1
    start = time.time()
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    frame_h, frame_w, _ = frame.shape
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_result = faceDetection.process(frameRGB)
    
    locs = []
    
    if detection_result.detections:
        for detection in detection_result.detections:
            relativeXMin = detection.location_data.relative_bounding_box.xmin
            relativeYMin = detection.location_data.relative_bounding_box.ymin
            relativeWidth = detection.location_data.relative_bounding_box.width
            relativeHeight = detection.location_data.relative_bounding_box.height
            
            xMin = min(math.floor(relativeXMin * frame_w), frame_w - 1)
            yMin = min(math.floor(relativeYMin * frame_h), frame_h - 1)
            xMax = min(math.floor((relativeXMin + relativeWidth) * frame_w), frame_w - 1)
            yMax = min(math.floor((relativeYMin + relativeHeight) * frame_h), frame_h - 1)
            locs.append((xMin, yMin, xMax, yMax))
            
            face = frameRGB[yMin:yMax, xMin:xMax]
            if face.any():
                face = cv2.resize(face, (224, 224))
                face = face.astype(np.float32) / 255.0
                face = np.expand_dims(face, 0)
            
                output_data = faceModel.predict(face)
                classification_result = np.squeeze(output_data)
                
                covered, uncovered = classification_result
                label = "incorrect" if uncovered > covered else "correct"
                percentage = uncovered if uncovered > covered else covered
                cv2.putText(frame, f"Mask: {label} ({percentage * 100:.2f}%)", (xMin - 40, yMin - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0) if label == "correct" else (0, 0, 255), 2)
                cv2.rectangle(frame, (xMin, yMin),
                                (xMax, yMax), (0, 255, 0) if label == "correct" else (0, 0, 255), 2)
    end = time.time()
    currentFPS = 1 / (end - start + 1e-12) # prevent zero time
    avgFPS = (1 - 1 / frameCount) * avgFPS + 1 / frameCount * currentFPS

    # show the output frame
    cv2.putText(frame, "Current FPS: " + str(round(currentFPS, 1)), (30, frame_h - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    cv2.putText(frame, "Average FPS: " + str(round(avgFPS, 1)), (30, frame_h - 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
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
