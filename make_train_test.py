import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", type=str)
ap.add_argument("--output_dir", type=str)
args = vars(ap.parse_args())

if os.path.exists(os.path.join(args["input_dir"], '.DS_Store')):
    os.remove(os.path.join(args["input_dir"], '.DS_Store'))

if os.path.exists(args["output_dir"]):
    shutil.rmtree(args["output_dir"])

print("Detect {} classes: {}".format(
    len(os.listdir(args["input_dir"])), os.listdir(args["input_dir"])))

images = []
labels = []

for label in os.listdir(args["input_dir"]):
    os.makedirs(os.path.join(args["output_dir"], "train", label))
    os.makedirs(os.path.join(args["output_dir"], "val", label))
    os.makedirs(os.path.join(args["output_dir"], "test", label))
    for imageName in os.listdir(os.path.join(args["input_dir"], label)):
        if imageName != ".DS_Store":
            images.append(imageName)
            labels.append(label)


imageTrain, imageTest, labelTrain, labelTest = train_test_split(
    images, labels, test_size=0.7)
imageTrain, imageVal, labelTrain, labelVal = train_test_split(
    imageTrain, labelTrain, test_size=0.5)

for image, label in tqdm(zip(imageTrain, labelTrain), total=len(imageTrain)):
    shutil.copyfile(os.path.join(
        args["input_dir"], label, image), os.path.join(args["output_dir"], "train", label, image))

for image, label in tqdm(zip(imageTest, labelTest), total=len(imageTest)):
    shutil.copyfile(os.path.join(
        args["input_dir"], label, image), os.path.join(args["output_dir"], "test", label, image))

for image, label in tqdm(zip(imageVal, labelVal), total=len(imageVal)):
    shutil.copyfile(os.path.join(
        args["input_dir"], label, image), os.path.join(args["output_dir"], "val", label, image))
