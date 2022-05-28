import argparse
import os
from tqdm import tqdm
import json


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str)
parser.add_argument("--input_dir", type=str)
parser.add_argument("--label_dir", type=str)
args = parser.parse_args()

if ".DS_Store" in os.listdir(args.input_dir):
    os.remove(os.path.join(args.input_dir, ".DS_Store"))

labels = os.listdir(args.input_dir)

images = []
annotations = []
categories = []

print(f"[INFO] - Detect {len(labels)} classes from {args.input_dir}")

obj_count = 0

for labelId, label in enumerate(labels):
    categories.append(dict(
        id=labelId,
        name=label,
        supercategory="mask"
    ))

yolo_labels = os.listdir(args.label_dir)

for imageId, yolo_label in tqdm(enumerate(yolo_labels)):
    images.append(dict(
        id=imageId,
        width=1024,
        height=1024,
        filename=yolo_label.split(".")[0] + ".jpg" 
    ))
    with open(os.path.join(args.label_dir, yolo_label)) as f:
        for line in f:
            obj_class, relativeCenterX, relativeCenterY, relativeWidth, relativeHeight = list(map(float, line.strip().split(" ")))
            
            # convert to absolute measure
            width = int(relativeWidth * 1024)
            height = int(relativeHeight * 1024)
            centerX = int(relativeCenterX * 1024)
            centerY = int(relativeCenterY * 1024)
            xMin = int(centerX - width / 2)
            yMin = int(centerY - height / 2)
            
            annotations.append(dict(
                id=obj_count,
                image_id = imageId,
                category_id=0 if "Mouth" in yolo_label or "Chin" in yolo_label or "Nose" in yolo_label else 1,
                segmentation=[],
                bbox=[xMin, yMin, width, height],
                area = width * height,
                iscrowd=0,
            ))
            
            obj_count += 1
                
coco_obj = dict(
    images = images,
    annotations = annotations,
    categories = categories
)

json.dump(coco_obj, open(args.output, "w"), indent=4)
            