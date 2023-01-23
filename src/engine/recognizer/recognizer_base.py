"""Recognizer base/parent class."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from typing import Union, List
from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from src.schema.face_recognition_schema import EmbeddingGTSchema, FaceRecognitionSchema
from src.utils.exception import APIExceptions
from src.utils.math import find_cosine_distance
from src.utils.logger import get_logger

exception = APIExceptions()
log = get_logger("recognizer_base")


class FaceRecognizerBase(ABC):
    """Face recognizer or encoder base/parent class."""

    def __init__(
        self,
        model_name: str,
        model_version: str,
        protocol: str,
        host: str,
        grpc_port: int,
        http_port: int,
    ) -> None:
        """Initialize."""
        assert protocol in ["grpc", "http"], "Protocol must be either 'grpc' or 'http'"
        self.model_name = model_name
        self.model_version = model_version
        self.protocol = protocol
        self.host = host
        self.grpc_port = grpc_port
        self.http_port = http_port
        self.input_shape = None

        # setup connection
        self.setup_connection()

        # get inputs and outputs from metadata
        self.inputs, self.outputs = self.get_matadata_io()

    def setup_connection(self) -> None:
        """Connect to OVMS server."""
        if self.protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(
                url=f"{self.host}:{self.grpc_port}", verbose=False
            )
            self.infer_input = grpcclient.InferInput
            self.infer_output = grpcclient.InferRequestedOutput
            log.log(
                24,
                f"{self.model_name.capitalize()} gRPC client connected to {self.host}:{self.grpc_port}",
            )
        elif self.protocol == "http":
            self.client = httpclient.InferenceServerClient(
                url=f"{self.host}:{self.http_port}", verbose=False
            )
            self.infer_input = httpclient.InferInput
            self.infer_output = httpclient.InferRequestedOutput
            log.log(
                24,
                f"{self.model_name.capitalize()} HTTP client connected to {self.host}:{self.http_port}",
            )
        else:
            raise exception.NotImplemented(f"Protocol {self.protocol} not implemented")

    def predict(
        self,
        face: np.ndarray,
        ground_truths: List[EmbeddingGTSchema],
        dist_method: str = "cosine",
        dist_threshold: float = 0.5,
    ) -> FaceRecognitionSchema:
        """
        Predict name of the person in the face image.

        Args:
            face (np.ndarray): Face image.
            ground_truths (List[EmbeddingGTSchema], optional): Ground truths. Defaults to None.
            dist_method (str, optional): Distance method. Defaults to "cosine".
            dist_threshold (float, optional): Distance threshold. Defaults to 0.5.

        Returns:
            FaceRecognitionSchema: Face recognition schema.
        """

        face_embedding = self.get_embedding(face)

        # find nearest neighbor
        dist = []
        for gt in ground_truths:
            if dist_method == "cosine":
                dist.append(find_cosine_distance(face_embedding, gt.embedding))
        min_dist = min(dist)
        min_dist_idx = dist.index(min_dist)
        if min_dist > dist_threshold:
            raise exception.NotFound("Face not matched with any face in database")
        return FaceRecognitionSchema(
            face=face,
            name=ground_truths[min_dist_idx].name,
            distance=min_dist,
            dist_method=dist_method,
        )

    def inference(
        self, face: np.ndarray
    ) -> Union[grpcclient.InferResult, httpclient.InferResult]:
        """
        Inference model with ovms client.

        Args:
            face (np.ndarray): Face image.

        Returns:
            Union[grpcclient.InferResult, httpclient.InferResult]: Inference result.
        """
        inputs = []
        inputs.append(self.infer_input(self.inputs, face.shape, "FP32"))
        inputs[0].set_data_from_numpy(face)
        outputs = []
        outputs.append(self.infer_output(self.outputs[0]))
        response = self.client.infer(
            self.model_name,
            inputs=inputs,
            model_version=self.model_version,
            outputs=outputs,
        )
        return response.as_numpy(self.outputs)

    def get_embedding(self, face: np.ndarray) -> Union[List[float], None]:
        """
        Get embedding from face image.

        Args:
            face (np.ndarray): Face image.

        Returns:
            Union[List[float], None]: Embedding vector.
        """
        face = self.preprocess_img(face)
        return self.inference(face)[0].tolist()

    def get_matadata_io(self) -> tuple:
        """Get inputs and outputs from metadata."""
        metadata = self.client.get_model_metadata(
            model_name=self.model_name, model_version=self.model_version
        )
        if self.protocol == "grpc":
            inputs = metadata.inputs[0].name
            outputs = metadata.outputs[0].name
        elif self.protocol == "http":
            inputs = metadata["inputs"][0]["name"]
            outputs = metadata["outputs"][0]["name"]

        return inputs, outputs

    def preprocess_img(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess image.

        Args:
            face (np.ndarray): Face image.

        Returns:
            np.ndarray: Preprocessed image.
        """
        face = cv2.resize(face, self.input_shape)
        face = face.transpose((2, 0, 1))
        face = np.expand_dims(face, axis=0)
        face = face.astype(np.float32)
        return face
