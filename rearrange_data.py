import os
import shutil
from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", type=str)
args = ap.parse_args()

BASE_DIR = args.data_dir

for folder in tqdm(os.listdir(BASE_DIR)):
    sub_image_dir = os.path.join(BASE_DIR, folder)
    if os.path.isdir(sub_image_dir):
        for image in os.listdir(sub_image_dir):
            shutil.move(os.path.join(sub_image_dir, image),
                        os.path.join(BASE_DIR, image))
        os.rmdir(sub_image_dir)
