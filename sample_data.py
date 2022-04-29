import argparse
import os
import random
import shutil
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", type=str)
ap.add_argument("--output_dir", type=str)
args = vars(ap.parse_args())

images = os.listdir(args["input_dir"])
if ".DS_Store" in images:
    images.remove(".DS_Store")

samples = random.sample(images, 10000)

if not os.path.exists(args["output_dir"]):
    os.makedirs(args["output_dir"])

for image in tqdm(samples):
    shutil.copy(os.path.join(args["input_dir"], image),
                os.path.join(args["output_dir"], image))
