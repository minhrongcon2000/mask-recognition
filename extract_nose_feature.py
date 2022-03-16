from utils import detect_face, extract_feature
import argparse
import cv2
import os
from tqdm import tqdm
import dlib

parser = argparse.ArgumentParser()
# parser.add_argument("--img_path", type=str)
parser.add_argument("--face_detector_path", type=str, default="face_detector")
parser.add_argument("--face_feature_extractor", type=str,
                    default="shape_predictor_68_face_landmarks.dat")
parser.add_argument("--output_dir", type=str, default="out")
parser.add_argument("--input_dir", type=str, default="dataset/CMFD")
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
# IMG_PATH = args.img_path
FACE_DETECTOR_PATH = args.face_detector_path
FACE_FEATURE_EXTRACTOR_PATH = args.face_feature_extractor
MIN_CONFIDENCE = 0.5

if os.path.exists(os.path.join(INPUT_DIR, ".DS_Store")):
    os.remove(os.path.join(INPUT_DIR, ".DS_Store"))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

architectPath = os.path.join(FACE_DETECTOR_PATH, "deploy.prototxt")
weightsPath = os.path.join(
    FACE_DETECTOR_PATH, "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(architectPath, weightsPath)
feature_extractor = dlib.shape_predictor(args.face_feature_extractor)

for IMG_PATH in tqdm(os.listdir(INPUT_DIR)):
    img = cv2.imread(os.path.join(INPUT_DIR, IMG_PATH))

    locs = detect_face(img, faceNet, return_confident=False)

    for i, box in enumerate(locs):
        (startX, startY, endX, endY) = box
        color = (0, 255, 0)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_feature = feature_extractor(
            gray_img, dlib.rectangle(startX, startY, endX, endY))

        noseStartX, noseStartY, noseEndX, noseEndY = extract_feature(
            face_feature, (27, 35))

        output_img_dir = os.path.join(OUTPUT_DIR, f"nose_{IMG_PATH}_{i}.png")

        nose_img = img[noseStartX:noseEndY, noseStartY:noseEndX]

        if nose_img.any() and (nose_img.shape[0] > nose_img.shape[1]):
            cv2.imwrite(output_img_dir, nose_img)
