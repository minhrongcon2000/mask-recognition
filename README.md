# My thesis: Face mask detection

## Introduction

Masks have became one of the most effective tools to fight against the spread of COVID-19 virus. However, people usually have a tendency to wear it in an incorrect way, degrading the efficiency of a mask. As a result, this thesis aims to solve this problem by developing a two-stage approach with the accuracy of 99% on the [MaskedFace-net dataset](https://github.com/cabani/MaskedFace-Net).

## Problem formulation

This thesis aims to solve a classification of mask usage behavior. A wrong usage is the use of mask that leave the nose or mouth open to the environment while a right usage covers these parts of body. As a result, the case of not wearing mask is a wrong usage of mask. As a result, instead of treating the problem as tri-ary classification. We only needs to treat it as binary classification.

## Methodology

This thesis uses a two-stage approach. The reason behind this is three-fold:

1. Every one-stage approach is expensive to train and expensive to inference. To my best understanding, the only model that seems to kinda solve this problem is RetinaNet, but it still has high cost of training.

2. Dataset for face mask detection is scarce. Remember, to make an object detection algorithm work well, the number of objects per image should be sufficiently large to capture all possible real-world scenario. Out of all dataset that I found on the internet, here are some datasets related to this problem:

    - [Face mask detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection): This dataset is the closest to the criteria of mask behavior recognition. However, this dataset is too scarce (only contains 853 images)

    - [Face mask detection dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset): This dataset is only for classification since it contains only one face per image

    - [MAFA](https://www.kaggle.com/datasets/rahulmangalampalli/mafa-data): This dataset seems good at first glance but only the test dataset seems to be satisfied with the criteria of dense object; however, the number of images and the density is not high enough as in COCO dataset.

    - [Face Mask Detection ~12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset): only for classification

    - [COVID-19 Medical Face Mask Detection Dataset](https://www.kaggle.com/datasets/mloey1/medical-face-mask-detection-dataset): for detection, but the dataset is scarce and needs a lot of relabelling.

3. Face detection has been around for a long time and even before deep learning era. As a result, face detection has long been solved with high AP and low inference time by a lot of deep learning models. As a result, if the face detection model is sub-millisecond fast, then the time it takes for classification model would be more forgiving. This is the case of [BlazeFace](https://arxiv.org/abs/1907.05047), implemented in [MediaPipe](https://google.github.io/mediapipe/solutions/face_detection).

With three reasons mentioned above, the methodology in this thesis follows the two-step process: first detecting face and second classifying the mask status on each extracted face in an image. For face detection, I use tools provided by [MediaPipe](https://google.github.io/mediapipe/solutions/face_detection). For classification, I trained a CNN model on the [MaskedFace-net dataset](https://github.com/cabani/MaskedFace-Net). This distribution of the training set, validation set, and testing set is shown in [here](https://www.kaggle.com/datasets/minhrongcon2000/maskedface-net).

## Results

Below are the results of three models, ResNet50, MobileNetV1, and VGG16 (both are trained with convolutional part frozen):

| Model           | Accuracy   | Inference time (unoptimized) |
|-----------------|------------|------------------------------|
| **MobileNetV1** | **99.39%** | **$0.1261\pm 0.0531$**       |
| VGG16           |     98.73% | $0.3436\pm 0.0722$           |
| ResNet50        |     50.07% | $0.2675\pm 0.2886$           |

As can be seen, MobileNetV1 is the best model that can be implemented on a limited resource device. Note: Inference time is benchmarked by using a Macbook Pro Mid-2015 version and the model here is not optimized by TFLite.

Video results on Raspberry Pi 4 Model B with LAN connection to the Macbook is shown [here](https://youtu.be/rQUpKr6K7Vc).

## Execute this repository

### Installation

Run `python3 -m pip install -r requirements.txt`

### Single device (image capturing and inferencing on one device)

Run `python3 detect_mediapipe.py`

### Client-server setup (image capturing on the client, image inferencing on the server)

On the server, run `python3 server.py` and wait for the terminal to display `Server started...`

On the client, run `python3 client.py --server_IP tcp://<your server IP>:5555` and see the result.

The expected result heavily depends on the connection between the client and the server. As a result, it is best to use a LAN cable for stable performance.

## Conclusion

In this thesis, I have shown the power of transfer learning and domain translation (specifically from synthetic image to real-world application). I also deployed it with high framerate on a Raspberry Pi 4 Model B.

## Future works

Maybe trying out:

- Using different part to predict the mask status: I'm actually fond of this idea but I don't have a good benchmark to show this is better than normal convolution

- Using one-stage: the idea of FPN and RetinaNet is really suitable for small object detection and is good for the problem of supervising mask usage in real-time (according to [RetinaFace](https://arxiv.org/pdf/1905.00641.pdf), the FPS is 16 on an ARM architecture)

- Aggregating more diverse data for one-stage approach: well, this may be a good next step, as the amount of data for this problem in the detection context is scarce and needs a lot of aggregation.
