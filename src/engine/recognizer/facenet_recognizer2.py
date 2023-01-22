"""Client for FaceNet recognizer."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from typing import Union
import numpy as np

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from src.utils.logger import get_logger

log = get_logger("facenet_ovms")


class FaceNetRecognizer:
    """FaceNet OVMS client."""

    def __init__(
        self,
        model_name: str = "facenet",
        model_version: str = "1",
        protocol: str = "grpc",
        host: str = "localhost",
        grpc_port: int = 4551,
        http_port: int = 4552,
    ) -> None:
        """
        Initialize.

        Args:
            model_version (str, optional): Model version. Defaults to "1".
            protocol (str, optional): Protocol to use. Defaults to "grpc".
        """
        assert protocol in ["grpc", "http"], "Protocol must be either 'grpc' or 'http'"
        self.model_name = model_name
        self.model_version = model_version
        self.protocol = protocol
        self.host = host
        self.grpc_port = grpc_port
        self.http_port = http_port

        # connect to server
        self.connect()

        # get inputs and outputs from metadata
        self.inputs, self.outputs = self.get_metadata_io()

    def inference(
        self, image: np.ndarray
    ) -> Union[grpcclient.InferResult, httpclient.InferResult]:
        """Inference."""
        inputs = []
        inputs.append(self.infer_input(self.inputs, image.shape, "FP32"))
        inputs[0].set_data_from_numpy(image)
        outputs = []
        outputs.append(self.infer_output(self.outputs))
        response = self.client.infer(
            model_name=self.model_name, inputs=inputs, outputs=outputs
        )
        return response.as_numpy(self.outputs)

    def connect(self) -> None:
        """Connect to OVMS server."""
        if self.protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(
                url=f"{self.host}:{self.grpc_port}", verbose=False
            )
            self.infer_input = grpcclient.InferInput
            self.infer_output = grpcclient.InferRequestedOutput
            log.log(
                24, f"FaceNet connected to gRPC server at {self.host}:{self.grpc_port}"
            )
        elif self.protocol == "http":
            self.client = httpclient.InferenceServerClient(
                url=f"{self.host}:{self.http_port}", verbose=False
            )
            self.infer_input = httpclient.InferInput
            self.infer_output = httpclient.InferRequestedOutput
            log.log(
                24, f"FaceNet connected to HTTP server at {self.host}:{self.http_port}"
            )

    def get_metadata_io(self) -> tuple:
        """Get input and output metadata."""
        metadata = self.client.get_model_metadata(
            model_name="facenet", model_version=self.model_version
        )
        if self.protocol == "grpc":
            inputs = metadata.inputs[0].name
            outputs = metadata.outputs[0].name
        elif self.protocol == "http":
            inputs = metadata["inputs"][0]["name"]
            outputs = metadata["outputs"][0]["name"]

        return inputs, outputs


if __name__ == "__main__":
    """Debugging."""

    import numpy as np
    import cv2
    import time

    image = cv2.imread("tmp/sample001_cropped.jpg")
    image = cv2.resize(image, (160, 160))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    facenet = FaceNetRecognizer(protocol="grpc")

    start = time.time()
    response = facenet.inference(image)
    end = time.time()

    print(f"Response time: {(end - start) * 1000:.2f} ms")
    print(response)
