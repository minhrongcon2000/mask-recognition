import random
import cv2
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", type=str)
args = parser.parse_args()

coco_info = json.load(open(args.label_dir))
images = coco_info["images"]
annotations = coco_info["annotations"]
categories = coco_info["categories"]

chosen_image = random.choice(images)
chosen_image_dir = None
if "Mouth" in chosen_image["filename"] or "Chin" in chosen_image["filename"] or "Nose" in chosen_image["filename"]:
    chosen_image_dir = "dataset/maskedface_net/IMFD/{}".format(chosen_image["filename"])
else:
    chosen_image_dir = "dataset/maskedface_net/CMFD/{}".format(chosen_image["filename"])

chosen_labels = filter(lambda item: item["image_id"] == chosen_image["id"], annotations)

for chosen_label in chosen_labels:
    x, y, box_w, box_h = chosen_label["bbox"]
    label = chosen_label["category_id"]
    img = cv2.imread(os.path.join(chosen_image_dir))
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 255, 0) if label == 1 else (0, 0, 255), 2)
    cv2.putText(img, "correct" if label == 1 else "incorrect", (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, color=(0, 255, 0) if label == 1 else (0, 0, 255))
cv2.imshow("output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
