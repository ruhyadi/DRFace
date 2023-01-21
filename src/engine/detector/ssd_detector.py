"""Face detector based on SSD model."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.engine.detector.detector_base import FaceDetectorBase
from src.schema.face_detection_schema import FaceDetectionSchema
from src.utils.logger import get_logger

log = get_logger("ssd_detector")


class SSDFaceDetector(FaceDetectorBase):
    """SSD Face detector module."""

    def __init__(
        self, model_dir: str = f"{ROOT}/weights/ssd_detector", min_conf: float = 0.5
    ) -> None:
        """
        Initialize SSD detector.

        Args:
            model_dir (str): Path to model directory. Defaults to f"{ROOT}/weights/ssd_detector".
            min_conf (float): Minimum confidence to detect a face. Defaults to 0.5.
        """
        self.model_dir = model_dir
        self.min_conf = min_conf
        self.model_weight = Path(self.model_dir) / "drface_ssd_v1.caffemodel"
        self.prototxt = Path(self.model_dir) / "drface_ssd_v1.prototxt"

    def build_model(self) -> None:
        """Build ssd model."""
        # check if model exists
        if not self.model_weight.exists() or not self.prototxt.exists():
            raise FileNotFoundError(
                "Model not found. Please download the model from github."
            )

        self.net = cv2.dnn.readNetFromCaffe(str(self.prototxt), str(self.model_weight))

    def detect_face(self, img: np.ndarray, align: bool = True) -> list[FaceDetectionSchema]:
        """
        Detect face from image.

        Args:
            img (np.ndarray): Image to detect face.
            align (bool, optional): Align face. Defaults to True.
        
        Returns:
            list[FaceDetectionSchema]: List of detected face.
        """
        h, w, c = img.shape

        # convert image to blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )

        # pass blob through model
        self.net.setInput(blob)
        detections = self.net.forward()

        # check for detections
        if len(detections) == 0:
            log.warning("No face detected")
            return []

        results = []
        for i in range(detections.shape[2]):
            score = detections[0, 0, i, 2]
            if score < self.min_conf:
                continue

            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = bbox.astype("int")

            # create face detection schema
            face = FaceDetectionSchema(
                face=img[y:y2, x:x2],
                coordinate=(x, y, x2, y2),
                score=score,
            )

            results.append(face)

        return results

if __name__ == "__main__":
    """Debugging."""

    img_path = f"{ROOT}/tmp/sample001.jpg"
    img = cv2.imread(img_path)
    detector = SSDFaceDetector()
    detector.build_model()

    faces = detector.detect_face(img)
    for face in faces:
        cv2.imwrite(f"{ROOT}/tmp/face_{face.score}.jpg", face.face)
        log.info(f"Face detected with score {face.score}")
        