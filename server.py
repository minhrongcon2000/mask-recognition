import json
import cv2
import imagezmq
import mediapipe as mp
import tensorflow.lite as tflite
import numpy as np
import math

mpFaceDetection = mp.solutions.face_detection

faceDetection = mpFaceDetection.FaceDetection()

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

imagehub = imagezmq.ImageHub()
print("Server started...")
while True:
    rpi_name, frame = imagehub.recv_image()
    frame_h, frame_w, _ = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_result = faceDetection.process(frameRGB)
    responses = []
    if detection_result.detections:
        for detection in detection_result.detections:
            relativeXMin = detection.location_data.relative_bounding_box.xmin
            relativeYMin = detection.location_data.relative_bounding_box.ymin
            relativeWidth = detection.location_data.relative_bounding_box.width
            relativeHeight = detection.location_data.relative_bounding_box.height
            
            xMin = min(math.floor(relativeXMin * frame_w), frame_w - 1)
            yMin = min(math.floor(relativeYMin * frame_h), frame_h - 1)
            xMax = min(math.floor((relativeXMin + relativeWidth) * frame_w), frame_w - 1)
            yMax = min(math.floor((relativeYMin + relativeHeight) * frame_h), frame_h - 1)
            
            face = frameRGB[yMin:yMax, xMin:xMax]
            if face.any():
                face = cv2.resize(face, (224, 224))
                face = face.astype(np.float32) / 255.0
                face = np.expand_dims(face, 0)
                
                interpreter.set_tensor(input_details[0]['index'], face)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                classification_result = np.squeeze(output_data)
                
                covered, uncovered = classification_result
                label = "incorrect" if uncovered > covered else "correct"
                percentage = uncovered if uncovered > covered else covered
                responses.append(dict(
                    xMin=xMin,
                    xMax=xMax,
                    yMin=yMin,
                    yMax=yMax,
                    label=label,
                    confidence=float(percentage),
                ))
    imagehub.send_reply(json.dumps(responses).encode("utf-8"))