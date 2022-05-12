import random
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str)
parser.add_argument("--label_dir", type=str)
args = parser.parse_args()

if os.path.isfile(os.path.join(args.source_dir, ".DS_Store")):
    os.remove(os.path.join(args.source_dir, ".DS_Store"))
images = os.listdir(args.source_dir)
chosen_image = random.choice(images)
chosen_label = open(os.path.join(args.label_dir, chosen_image.split(".")[0] + ".txt")).read().split(" ")
label, x, y, box_w, box_h = list(map(float, chosen_label))
img = cv2.imread(os.path.join(args.source_dir, chosen_image))
h, w, _ = img.shape
realX = x * w
realY = y * h
realBoxW = int(box_w * w)
realBoxH = int(box_h * h)
startX = int(realX - realBoxW / 2)
startY = int(realY - realBoxH / 2)
cv2.rectangle(img, (startX, startY), (startX + realBoxW,
              startY + realBoxH), (0, 255, 0), 2)
cv2.imshow("output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
