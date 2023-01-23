"""Face detection model/schema."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from dataclasses import dataclass

import numpy as np
from bson import ObjectId
from fastapi import Form
from pydantic import BaseModel, Field

from src.schema.enums_schema import DetectionModel, RecognitionModel


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


@dataclass
class FaceDetectionRequest:
    """Face detection API request schema."""

    name: str = Form(..., description="Name of the person")
    detection_model: DetectionModel = Form(..., description="Detection model")
    recognition_model: RecognitionModel = Form(..., description="Recognition model")


class FaceDetectionResponse(BaseModel):
    """Face detection API response schema."""

    request_id: str = Field(None, description="Request ID")
    timestamp: str = Field(None, description="Timestamp")
    status: str = Field(None, description="Status")
    detection_model: DetectionModel = Field(
        DetectionModel.ssd, description="Face detection model"
    )
    recognition_model: RecognitionModel = Field(
        RecognitionModel.arcface, description="Face recognition model"
    )
    name: str = Field(None, description="Face name")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "request_id": "5f9f5b5b-5f9f5b5b-5f9f5b5b-5f9f5b5b",
                "timestamp": "2021-01-03T00:00:01.000000Z",
                "status": "success",
                "detection_model": "ssd",
                "recognition_model": "arcface",
                "name": "didi_ruhyadi",
            }
        }
