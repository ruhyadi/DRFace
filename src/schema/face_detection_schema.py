"""Face detection model/schema."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np


class FaceDetectionSchema:
    """Face detection schema."""

    def __init__(
        self, face: np.ndarray, coordinate: list, score: float, landmarks: list = None
    ) -> None:
        """
        Initialize face detection schema.

        Args:
            face (np.ndarray): Face image.
            coordinate (list): Face coordinate. XYXY format.
            score (float, optional): Face detection score.
            landmarks (list, optional): Face landmarks. Defaults to None.
        """
        self.face = face
        self.coordinate = coordinate
        self.landmarks = landmarks
        self.score = score

    def xywh2xyxy(self) -> list:
        """Convert xywh to xyxy."""
        x, y, w, h = self.coordinate
        return [x, y, x + w, y + h]

    def xyxy2xywh(self) -> list:
        """Convert xyxy to xywh."""
        x1, y1, x2, y2 = self.coordinate
        return [x1, y1, x2 - x1, y2 - y1]
