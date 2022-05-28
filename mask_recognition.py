import config
import cv2
import numpy as np

from utils import detect_face, extract_feature

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(config.prototxtPath, config.weightsPath)


def make_inference_from_frame(frame):
    locs = detect_face(frame, faceNet)
    return np.array(locs)
