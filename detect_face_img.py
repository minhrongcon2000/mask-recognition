from utils import detect_face
import argparse
import cv2
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str)
parser.add_argument("--face_detector_path", type=str)
args = parser.parse_args()

IMG_PATH = args.img_path
FACE_DETECTOR_PATH = args.face_detector_path
MIN_CONFIDENCE = 0.5

architectPath = os.path.join(FACE_DETECTOR_PATH, "deploy.prototxt")
weightsPath = os.path.join(
    FACE_DETECTOR_PATH, "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(architectPath, weightsPath)

img = cv2.imread(IMG_PATH)

locs = detect_face(img, faceNet)

for box in locs:
    startX, startY, endX, endY = box
    color = (0, 255, 0)
    plt.imshow(img[startY:endY, startX:endX, :])
    plt.show()
    # cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)


# cv2.imshow("Output", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
