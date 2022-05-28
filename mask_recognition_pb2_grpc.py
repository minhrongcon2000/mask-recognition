# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import mask_recognition_pb2 as mask__recognition__pb2


class MaskRecognitionStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.make_inference_from_frame = channel.unary_unary(
                '/MaskRecognition/make_inference_from_frame',
                request_serializer=mask__recognition__pb2.B64Image.SerializeToString,
                response_deserializer=mask__recognition__pb2.Prediction.FromString,
                )


class MaskRecognitionServicer(object):
    """Missing associated documentation comment in .proto file."""

    def make_inference_from_frame(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MaskRecognitionServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'make_inference_from_frame': grpc.unary_unary_rpc_method_handler(
                    servicer.make_inference_from_frame,
                    request_deserializer=mask__recognition__pb2.B64Image.FromString,
                    response_serializer=mask__recognition__pb2.Prediction.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'MaskRecognition', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MaskRecognition(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def make_inference_from_frame(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/MaskRecognition/make_inference_from_frame',
            mask__recognition__pb2.B64Image.SerializeToString,
            mask__recognition__pb2.Prediction.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)