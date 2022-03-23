# USAGE
# python detect_mask_video.py

# import the necessary packages
from imutils.video import VideoStream
import imutils
import time
import cv2
import dlib

from utils import detect_face, extract_feature

# load pretrained face detection model
print("[INFO] loading face detector model...")
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# load face feature extractor
feature_extractor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    locs = detect_face(frame, faceNet)

    for box in locs:
        (startX, startY, endX, endY) = box

        color = (0, 255, 0)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_features = feature_extractor(
            gray_frame, dlib.rectangle(startX, startY, endX, endY))

        # FEATURE NEEDED
        # Jaw: (4, 12)
        # Mouth: (48, 67)
        # Nose: (27, 35)
        featureStartX, featureStartY, featureEndX, featureEndY = extract_feature(
            face_features, (4, 12))

        cv2.rectangle(frame, (featureStartX, featureStartY),
                      (featureEndX, featureEndY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
