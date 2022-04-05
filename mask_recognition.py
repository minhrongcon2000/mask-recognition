import config
import cv2
from tensorflow.keras.models import load_model
import dlib
import numpy as np

from utils import detect_face, extract_feature

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(config.prototxtPath, config.weightsPath)

# nose model
print("[INFO] - loading nose model...")
noseModel = load_model(config.noseModelPath)

# load face feature extractor, use for nose detection
feature_extractor = dlib.shape_predictor(config.featureExtractorPath)


def make_inference_from_frame(frame):
    locs = detect_face(frame, faceNet)

    noses = []
    nose_locs = []

    # extract nose from each face in the frame
    for box in locs:
        (startX, startY, endX, endY) = box

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_features = feature_extractor(
            gray_frame, dlib.rectangle(startX, startY, endX, endY))

        # nose part
        noseStartX, noseStartY, noseEndX, noseEndY = extract_feature(
            face_features, (27, 35))
        nose = frame[noseStartY:noseEndY, noseStartX:noseEndX]
        nose = cv2.resize(nose, (224, 224))
        nose = nose / 255.
        noses.append(nose)
        nose_locs.append((noseStartX, noseStartY, noseEndX, noseEndY))

    noses = np.array(noses)
    if noses.any():
        nosePredictions = noseModel.predict(noses)
        return np.array(locs), np.array(nose_locs), nosePredictions.astype(float)
    return None
