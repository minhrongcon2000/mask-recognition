import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", type=str)
ap.add_argument("--feature", type=str)
args = vars(ap.parse_args())

num_image = 0
images = os.listdir(args["input_dir"])
if ".DS_Store" in images:
    images.remove(".DS_Store")

num_sample = 0

for image in os.listdir(args["input_dir"]):
    if args["feature"] not in image.lower():
        num_sample += 1

print(num_sample, len(images))
