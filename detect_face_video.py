from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os
import dlib
import numpy as np

from utils import detect_face, extract_feature
import tflite_runtime.interpreter as tflite

import config

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(config.prototxtPath, config.weightsPath)

# nose model
print("[INFO] - loading nose model...")
noseModel = tflite.Interpreter(model_path=config.noseModelPath)
noseModel.allocate_tensors()
input_details = noseModel.get_input_details()
output_details = noseModel.get_output_details()

width = input_details[0]["shape"][1]
height = input_details[0]["shape"][2]


# load face feature extractor, use for nose detection
feature_extractor = dlib.shape_predictor(config.featureExtractorPath)

num_frames = 0  # counter for how many frames have passed
save_period = 50  # period of recognizing face and storing data


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(2.0)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

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
        nose = cv2.resize(nose, (width, height))
        nose = nose / 255.
        noses.append(nose)
        nose_locs.append((noseStartX, noseStartY, noseEndX, noseEndY))

    noses = np.array(noses)
    if noses.any():
        noseModel.set_tensor(input_details[0]['index'], noses)
        noseModel.invoke()
        nosePredictions = noseModel.get_tensor(output_details[0]['index'])
        for i, (noseStartX, noseStartY, noseEndX, noseEndY) in enumerate(nose_locs):
            covered, uncovered = nosePredictions[i]
            print(covered, uncovered)
            label = "Nose: {}".format(
                "uncovered" if covered <= uncovered else "covered")
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
