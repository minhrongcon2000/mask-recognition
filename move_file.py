import os
import shutil
from tqdm import tqdm
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", type=str)
ap.add_argument("--output_dir", type=str)
args = vars(ap.parse_args())

if os.path.exists(os.path.join(args["input_dir"], '.DS_Store')):
    os.remove(os.path.join(args["input_dir"], '.DS_Store'))

if not os.path.exists(args["output_dir"]):
    os.makedirs(args["output_dir"])

for image in tqdm(os.listdir(args["input_dir"])):
    shutil.move(os.path.join(args["input_dir"], image),
                os.path.join(args["output_dir"], image))
