from turtle import width
from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np
import grpc
import mask_recognition_pb2
import mask_recognition_pb2_grpc
from utils import decode_matrix_base64, encode_message_base64
import json

HOST_IP = "localhost"
PORT = "50051"

with grpc.insecure_channel(f'{HOST_IP}:{PORT}') as channel:
    stub = mask_recognition_pb2_grpc.MaskRecognitionStub(channel)

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        height, width, c = frame.shape
        encoded_frame = encode_message_base64(np.array(frame, dtype=np.uint8))

        request = mask_recognition_pb2.B64Image(
            b64image=encoded_frame, width=width, height=height, channel=c)

        results = stub.make_inference_from_frame(request)
        
        results = json.loads(results.b64facelocs)

        for result in results:
            startX = result['xMin']
            startY = result['yMin']
            endX = result['xMax']
            endY = result['yMax']
            label = "correct" if result['confidence']['correct'] > result['confidence']['incorrect'] else 'incorrect'
            percentage = max(result['confidence']['correct'], result['confidence']['incorrect'])
            cv2.putText(frame, f"Mask: {label} ({percentage * 100:.2f}%)", (startX - 40, startY - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0) if label == "correct" else (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY),
                        (endX, endY), (0, 255, 0) if label == "correct" else (0, 0, 255), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
