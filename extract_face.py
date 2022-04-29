from turtle import distance
from utils import detect_face, extract_feature
import argparse
import cv2
import os
from tqdm import tqdm
import dlib
import math

FEATURE2INDEX = {
    "nose": (27, 35),
    "mouth": (48, 67),
    "chin": (4, 12),
}

parser = argparse.ArgumentParser()
# parser.add_argument("--img_path", type=str)
parser.add_argument("--face_detector_path", type=str, default="face_detector")
parser.add_argument("--output_dir", type=str, default="out")
parser.add_argument("--input_dir", type=str, default="dataset/CMFD")
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
# IMG_PATH = args.img_path
FACE_DETECTOR_PATH = args.face_detector_path

if os.path.exists(os.path.join(INPUT_DIR, ".DS_Store")):
    os.remove(os.path.join(INPUT_DIR, ".DS_Store"))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

architectPath = os.path.join(FACE_DETECTOR_PATH, "deploy.prototxt")
weightsPath = os.path.join(
    FACE_DETECTOR_PATH, "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(architectPath, weightsPath)

for IMG_PATH in tqdm(os.listdir(INPUT_DIR)):
    img = cv2.imread(os.path.join(INPUT_DIR, IMG_PATH))

    locs = detect_face(img, faceNet)

    chosen_loc = None
    min_distance = None

    for loc in locs:
        (startX, startY, endX, endY) = loc
        centerX = (startX + endX) / 2
        centerY = (startY + endY) / 2

        distance = math.sqrt((centerX - 1024 / 2) ** 2 +
                             (centerY - 1024 / 2) ** 2)

        if min_distance is None or distance < min_distance:
            chosen_loc = loc
            min_distance = distance

    if chosen_loc is not None:
        startX, startY, endX, endY = chosen_loc
        output_img_dir = os.path.join(
            OUTPUT_DIR, f"{IMG_PATH}.png")
        feature_img = img[startY:endY, startX:endX]
        if feature_img.any():
            cv2.imwrite(output_img_dir, img[startY:endY, startX:endX])
