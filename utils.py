import cv2
import numpy as np
import base64


def detect_face(frame, faceNet, confidence_threshhold=0.5):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > confidence_threshhold:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return locs


def extract_feature(features, landmark_index):
    start, end = landmark_index
    coordinates = np.zeros((end - start + 1, 2), dtype=int)

    for i in range(start, end + 1):
        coordinates[i - start] = [features.part(i).x, features.part(i).y]

    return coordinates[:, 0].min(), coordinates[:, 1].min(), coordinates[:, 0].max(), coordinates[:, 1].max()


def encode_message_base64(img):
    return base64.b64encode(img)


def decode_image_base64(msg, width, height, channel):
    b64decoded_buffer = base64.b64decode(msg)
    img_arr = np.frombuffer(b64decoded_buffer).reshape(width, height, channel)
    return img_arr


def decode_matrix_base64(msg, shape, dtype=np.uint8):
    b64decoded_buffer = base64.b64decode(msg)
    matrix = np.frombuffer(b64decoded_buffer, dtype=dtype).reshape(*shape)
    return matrix
