# USAGE
# python detect_mask_video.py

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import dlib
import numpy as np

from utils import detect_face, extract_feature
from tensorflow.keras.models import load_model


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("--face_feature_extractor", type=str,
                default="shape_predictor_68_face_landmarks.dat")
args = vars(ap.parse_args())

# load pretrained face detection model
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
# IMPLEMENT LATER

# nose model
noseModel = load_model("nose_model/model-best.h5")

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# load face feature extractor
feature_extractor = dlib.shape_predictor(args["face_feature_extractor"])

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    locs = detect_face(frame, faceNet)
    noses = []

    # loop over the detected face locations and their corresponding
    # locations
    for box in locs:
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box

        # determine the class label and color we'll use to draw
        # the bounding box and text
        color = (0, 255, 0)

        # display the label and bounding box rectangle on the output
        # frame
        # cv2.putText(frame, label, (startX, startY - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_features = feature_extractor(
            gray_frame, dlib.rectangle(startX, startY, endX, endY))

        noseStartX, noseStartY, noseEndX, noseEndY = extract_feature(
            face_features, (27, 35))
        nose = frame[noseStartY:noseEndY, noseStartX:noseEndX]
        nose = cv2.resize(nose, (224, 224))
        nose = nose / 255.
        noses.append(nose)

        cv2.rectangle(frame, (noseStartX, noseStartY),
                      (noseEndX, noseEndY), color, 2)

    noses = np.array(noses)
    if noses.any():
        predictions = noseModel.predict(noses)
        print(predictions)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
