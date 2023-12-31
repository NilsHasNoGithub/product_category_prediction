# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import shoe_category_prediction_pb2 as shoe__category__prediction__pb2


class CategoryPredictionStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetImagePrediction = channel.unary_unary(
            "/CategoryPrediction/GetImagePrediction",
            request_serializer=shoe__category__prediction__pb2.ImagePredictionRequest.SerializeToString,
            response_deserializer=shoe__category__prediction__pb2.PredictionReply.FromString,
        )
        self.GetTextPrediction = channel.unary_unary(
            "/CategoryPrediction/GetTextPrediction",
            request_serializer=shoe__category__prediction__pb2.TextPredictionRequest.SerializeToString,
            response_deserializer=shoe__category__prediction__pb2.PredictionReply.FromString,
        )


class CategoryPredictionServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetImagePrediction(self, request, context):
        """Obtains a category prediction for a shoe image, encoded as bytes"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def GetTextPrediction(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_CategoryPredictionServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "GetImagePrediction": grpc.unary_unary_rpc_method_handler(
            servicer.GetImagePrediction,
            request_deserializer=shoe__category__prediction__pb2.ImagePredictionRequest.FromString,
            response_serializer=shoe__category__prediction__pb2.PredictionReply.SerializeToString,
        ),
        "GetTextPrediction": grpc.unary_unary_rpc_method_handler(
            servicer.GetTextPrediction,
            request_deserializer=shoe__category__prediction__pb2.TextPredictionRequest.FromString,
            response_serializer=shoe__category__prediction__pb2.PredictionReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "CategoryPrediction", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class CategoryPrediction(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetImagePrediction(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/CategoryPrediction/GetImagePrediction",
            shoe__category__prediction__pb2.ImagePredictionRequest.SerializeToString,
            shoe__category__prediction__pb2.PredictionReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def GetTextPrediction(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/CategoryPrediction/GetTextPrediction",
            shoe__category__prediction__pb2.TextPredictionRequest.SerializeToString,
            shoe__category__prediction__pb2.PredictionReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
