import os
from sklearn.model_selection import train_test_split
import shutil
from datetime import datetime

RIGHT_MASK_DIR = './dataset/right_mask'
NO_MASK_DIR = './dataset/without_mask'
WRONG_MASK_DIR = './dataset/wrong_mask'

WRONG_MASK_IMG = os.listdir(WRONG_MASK_DIR)
NO_MASK_IMG = os.listdir(NO_MASK_DIR)
RIGHT_MASK_IMG = os.listdir(RIGHT_MASK_DIR)

images = RIGHT_MASK_IMG + NO_MASK_IMG + WRONG_MASK_IMG
labels = ['right_mask' for _ in range(len(RIGHT_MASK_IMG))] + ['no_mask' for _ in range(
    len(NO_MASK_IMG))] + ['wrong_mask' for _ in range(len(WRONG_MASK_IMG))]

images_train, images_val, label_train, label_val = train_test_split(
    images, labels, test_size=0.2)

label2source = {
    "right_mask": RIGHT_MASK_DIR,
    "wrong_mask": WRONG_MASK_DIR,
    "no_mask": NO_MASK_DIR,
}

createTime = int(datetime.now().timestamp())
os.makedirs(f"datasets_{createTime}/train/right_mask")
os.makedirs(f"datasets_{createTime}/train/no_mask")
os.makedirs(f"datasets_{createTime}/train/wrong_mask")
os.makedirs(f"datasets_{createTime}/val/right_mask")
os.makedirs(f"datasets_{createTime}/val/no_mask")
os.makedirs(f"datasets_{createTime}/val/wrong_mask")

for imageName, label in zip(images_train, label_train):
    shutil.copyfile(
        os.path.join(label2source[label], imageName),
        os.path.join(f"datasets_{createTime}/train", label, imageName)
    )
for imageName, label in zip(images_val, label_val):
    shutil.copyfile(
        os.path.join(label2source[label], imageName),
        os.path.join(f"datasets_{createTime}/val", label, imageName)
    )
