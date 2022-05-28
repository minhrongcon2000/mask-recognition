import grpc
import numpy as np
import mask_recognition
import mask_recognition_pb2
import mask_recognition_pb2_grpc
from concurrent import futures
import json

from utils import decode_matrix_base64, encode_message_base64


class MaskRecognitionService(mask_recognition_pb2_grpc.MaskRecognitionServicer):
    def make_inference_from_frame(self, request, context):
        # print("[INFO] - Function called...")
        response = mask_recognition_pb2.Prediction()
        img_array = decode_matrix_base64(
            request.b64image, (request.height, request.width, request.channel))
        results = mask_recognition.make_inference_from_frame(img_array)
        response.b64facelocs = json.dumps(results)
        return response


def serve():
    print("[INFO] - Initialize server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mask_recognition_pb2_grpc.add_MaskRecognitionServicer_to_server(
        MaskRecognitionService(), server)
    server.add_insecure_port("[::]:50051")
    print("[INFO] - Server listen on port 50051...")
    server.start()
    server.wait_for_termination()


serve()
