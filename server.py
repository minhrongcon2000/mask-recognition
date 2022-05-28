from urllib import response
import grpc
import mask_recognition
import mask_recognition_pb2
import mask_recognition_pb2_grpc
from concurrent import futures

from utils import decode_matrix_base64, encode_message_base64


class MaskRecognitionService(mask_recognition_pb2_grpc.MaskRecognitionServicer):
    def make_inference_from_frame(self, request, context):
        print("[INFO] - Function called...")
        response = mask_recognition_pb2.Prediction()
        img_array = decode_matrix_base64(
            request.b64image, (request.width, request.height, request.channel))
        results = mask_recognition.make_inference_from_frame(img_array)
        if results is not None:
            face_locs = results
            response.status = 'successful'
            response.b64facelocs = encode_message_base64(face_locs)
        else:
            response.status = 'failed'
            response.b64facelocs = '0'
            response.b64noselocs = '0'
            response.b64nosepreds = '0'
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
