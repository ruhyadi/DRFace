"""Face recognition model/schema."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np


class FaceRecognitionSchema:
    """Face recognition schema."""

    def __init__(self, face: np.ndarray, name: str, distance: float, dist_method: str) -> None:
        """
        Initialize face recognition schema.

        Args:
            face (np.ndarray): Face image.
            name (str): Face name.
            distance (float): Face distance.
            dist_method (str): Distance method.
        """
        self.face = face
        self.name = name
        self.distance = distance
        self.dist_method = dist_method

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "distance": self.distance,
            "dist_method": self.dist_method,
        }


class EmbeddingGTSchema:
    """Embedding ground truth schema."""

    def __init__(self, name: str, embedding: np.ndarray) -> None:
        """
        Initialize embedding ground truth schema.

        Args:
            name (str): Face name.
            embedding (np.ndarray): Face embedding.
        """
        self.name = name
        self.embedding = embedding

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "embedding": self.embedding,
        }
