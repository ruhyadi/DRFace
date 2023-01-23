"""Recognizer base/parent class."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.schema.face_recognition_schema import EmbeddingGTSchema, FaceRecognitionSchema


class FaceRecognizerBase(ABC):
    """Face recognizer or encoder base/parent class."""

    @abstractmethod
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

    @abstractmethod
    def setup_connection(self) -> None:
        """Setup connection to server."""
        pass

    @abstractmethod
    def predict(
        self,
        face: np.ndarray,
        ground_truths: List[EmbeddingGTSchema] = None,
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
        pass

    @abstractmethod
    def get_embedding(self, face: np.ndarray) -> list:
        """
        Get face embedding.

        Args:
            face (np.ndarray): Face image.

        Returns:
            list: Face embedding vector.
        """
        pass
