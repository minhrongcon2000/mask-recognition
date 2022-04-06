import base64
from imutils.video import VideoStream
import imutils
import time
import cv2
import os
import numpy as np
import grpc
import mask_recognition_pb2
import mask_recognition_pb2_grpc
from utils import decode_matrix_base64, encode_message_base64

HOST_IP = "192.168.1.239"
PORT = "50051"

with grpc.insecure_channel(f'{HOST_IP}:{PORT}') as channel:
    stub = mask_recognition_pb2_grpc.MaskRecognitionStub(channel)

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        width, height, channel = frame.shape
        encoded_frame = encode_message_base64(np.array(frame, dtype=np.uint8))

        request = mask_recognition_pb2.B64Image(
            b64image=encoded_frame, width=width, height=height, channel=channel)

        results = stub.make_inference_from_frame(request)

        if results.status != "failed":
            face_locs = decode_matrix_base64(
                results.b64facelocs, (-1, 4), dtype=int)
            nose_locs = decode_matrix_base64(
                results.b64noselocs, (-1, 4), dtype=int)
            nose_preds = decode_matrix_base64(
                results.b64nosepreds, (-1, 2), dtype=float, preprocessing=False)

            for (startX, startY, endX, endY) in face_locs:
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)

            for nose_loc, nose_pred in zip(nose_locs, nose_preds):
                startX, startY, endX, endY = nose_loc
                covered, uncovered = nose_pred
                label = "Nose: {}".format(
                    "uncovered" if uncovered >= covered or abs(uncovered - covered) <= 0.2 else "covered")
                #print(covered, uncovered, label)
                cv2.putText(frame, label, (startX - 10, startY),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
