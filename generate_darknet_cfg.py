import argparse
import math
import os
import cv2
from tqdm import tqdm
from utils import detect_face


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--model_config", type=str)
parser.add_argument("--model_weights", type=str)
args = parser.parse_args()

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(args.model_config, args.model_weights)

if ".DS_Store" in os.listdir(args.input_dir):
    os.remove(os.path.join(args.input_dir, ".DS_Store"))
    
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

labels = os.listdir(args.input_dir)
print(f"[INFO] - Detect {len(labels)} classes from {args.input_dir}")

with open("yolo.names", "w") as f:
    f.write("\n".join(labels))
    
with open("yolo.data", "w") as f:
    f.write(f"classes = {len(labels)}\ntrain = train.txt\nval = val.txt\nbackup = backup")

for i, label in enumerate(labels):
    images = os.listdir(os.path.join(args.input_dir, label))
    print("[INFO] - Generate label for class {}...".format(label))
    for image_file in tqdm(images):
        img = cv2.imread(os.path.join(args.input_dir, label, image_file))
        h, w, _ = img.shape
        locs = detect_face(img, faceNet)

        chosen_loc = None
        min_dist = None

        for (startX, startY, endX, endY) in locs:
            centerX = (startX + endX) / 2
            centerY = (startY + endY) / 2
            distToCenter = math.sqrt(
                (centerX - w / 2) ** 2 + (centerY - h / 2) ** 2)
            if chosen_loc is None or distToCenter < min_dist:
                chosen_loc = (startX, startY, endX, endY)
                min_dist = distToCenter

        if chosen_loc is not None:
            startX, startY, endX, endY = chosen_loc
            centerX = (startX + endX) / (2 * w)
            centerY = (startY + endY) / (2 * h)
            boxWidth = (endX - startX) / w
            boxHeight = (endY - startY) / h
            open(os.path.join(args.output_dir, image_file.split(".")[0] + ".txt"), "w").write(f"{i} {centerX} {centerY} {boxWidth} {boxHeight}")
