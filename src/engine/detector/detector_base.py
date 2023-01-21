"""Face detector base/parent class."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from src.utils.math import find_euclidean_distance


class FaceDetectorBase(ABC):
    """Face detector base/parent class."""

    def __init__(self) -> None:
        """Initialize."""
        pass

    @abstractmethod
    def build_model(self) -> None:
        """Build face detector model."""
        pass

    @abstractmethod
    def detect_face(self, img: np.ndarray, align: bool = True) -> list:
        """
        Detect faces in an image.

        Args:
            img (np.ndarray): Image to detect faces in.
            align (bool): Whether to align faces.

        Returns:
            list: List of detected faces.
        """
        pass

    def alignment_procedure(
        self, face: np.ndarray, left_eye: tuple, right_eye: tuple
    ) -> np.ndarray:
        """
        Face alignment procedure.
        function aligns given face in img based on left and right eye coordinates.

        Args:
            face (np.ndarray): Detected face.
            left_eye (tuple): Left eye coordinates (x1, y1).
            right_eye (tuple): Right eye coordinates (x2, y2).

        Returns:
            np.ndarray: Aligned face.
        """
        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        # find rotation direction
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock

        # find length of triangle edges
        a = find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
        b = find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
        c = find_euclidean_distance(np.array(right_eye), np.array(left_eye))

        # apply cosine rule
        if (
            b != 0 and c != 0
        ):  # this multiplication causes division by zero in cos_a calculation
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)  # angle in radian
            angle = (angle * 180) / np.pi  # radian to degree

            # rotate base image
            if direction == -1:
                angle = 90 - angle

            # TODO: rotate image with numpy instead of PIL
            face = Image.fromarray(face)
            face = np.array(face.rotate(direction * angle))

        return face
