from imutils.video import VideoStream
import imutils
import time
import cv2
import mediapipe as mp
import math
import numpy as np
import tensorflow.lite as tflite

mpFaceDetection = mp.solutions.face_detection

faceDetection = mpFaceDetection.FaceDetection()

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

captureTime = []

while True:
    start = time.time()
    frame = vs.read()
    frame = imutils.resize(frame, width=1024)
    
    frame_h, frame_w, _ = frame.shape
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_result = faceDetection.process(frameRGB)
    
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
                cv2.putText(frame, f"Mask: {label} ({percentage * 100:.2f}%)", (xMin - 40, yMin - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0) if label == "correct" else (0, 0, 255), 2)
                cv2.rectangle(frame, (xMin, yMin),
                                (xMax, yMax), (0, 255, 0) if label == "correct" else (0, 0, 255), 2)
    end = time.time()
    captureTime.append(1 / (end - start))
    cv2.putText(frame, "FPS: " + str(round(1 / (end - start), 1)), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
print(np.array(captureTime).mean())
print(np.array(captureTime).std(ddof=1))
print(len(captureTime))
