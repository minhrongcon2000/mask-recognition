import config
import cv2
import numpy as np
import tensorflow as tf

from utils import detect_face

print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(config.prototxtPath, config.weightsPath)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classify_face(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.
    frame = np.expand_dims(frame, 0)
    interpreter.set_tensor(input_details[0]['index'], frame)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    return results


def make_inference_from_frame(frame):
    locs = detect_face(frame, faceNet)
    results = []
    for (xMin, yMin, xMax, yMax) in locs:
        face = frame[yMin:yMax, xMin:xMax].astype(np.float32)
        covered, uncovered = classify_face(face)
        results.append(dict(
            xMin=int(xMin),
            yMin=int(yMin),
            xMax=int(xMax),
            yMax=int(yMax),
            confidence={
                "correct": float(covered),
                "incorrect": float(uncovered)
            }
        ))
    return results
