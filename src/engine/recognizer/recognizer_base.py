"""Recognizer base/parent class."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from abc import ABC, abstractmethod

import numpy as np


class FaceRecognizerBase(ABC):
    """Face recognizer or encoder base/parent class."""
    def __init__(self) -> None:
        """Initialize."""
        pass

    @abstractmethod 
    def build_model(self) -> None:
        """Build face recognizer model."""
        pass

    @abstractmethod
    def predict(self, img: np.ndarray) -> np.ndarray:
        """
        Predict face embedding.

        Args:
            img (np.ndarray): Image to predict face embedding from.

        Returns:
            np.ndarray: Face embedding.
        """
        pass

    @abstractmethod
    def get_embeddings(self, img: np.ndarray) -> np.ndarray:
        """
        Get face embeddings.

        Args:
            img (np.ndarray): Image to get face embeddings from.

        Returns:
            np.ndarray: Face embeddings.
        """
        pass