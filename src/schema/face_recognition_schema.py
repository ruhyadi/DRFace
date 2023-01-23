"""Face recognition model/schema."""

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

from src.schema.common_schema import PyObjectId
from src.schema.enums_schema import DetectionModel, RecognitionModel


class FaceRecognitionSchema:
    """Face recognition schema."""

    def __init__(
        self, face: np.ndarray, name: str, distance: float, dist_method: str
    ) -> None:
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


class FaceEmbeddingSchema(BaseModel):
    """Face embedding schema."""

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId = Field(None, description="User ID")
    name: str = Field(None, description="Face name")
    detection_model: DetectionModel = Field(None, description="Detection model")
    recognition_model: RecognitionModel = Field(None, description="Recognition model")
    embedding: list = Field(..., description="Face embedding")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


@dataclass
class FaceRecognitionRequest:
    """Face recognition API request schema."""

    detection_model: DetectionModel = Form(..., description="Detection model")
    recognition_model: RecognitionModel = Form(..., description="Recognition model")


class FaceRecognitionResponse(BaseModel):
    """Face recognition API response schema."""

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    request_id: str = Field(None, description="Request ID")
    timestamp: str = Field(None, description="Timestamp")
    status: str = Field(None, description="Status")
    detection_model: DetectionModel = Field(..., description="Detection model")
    recognition_model: RecognitionModel = Field(..., description="Recognition model")
    name: str = Field(None, description="Face name")
    distance: float = Field(None, description="Face distance")
    dist_method: str = Field(None, description="Distance method")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        sceham_extra = {
            "example": {
                "request_id": "5f9f5b9b-5b5a-4b5a-9b9b-5b9b5b9b5b9b",
                "timestamp": "2021-01-03T00:00:01.000000Z",
                "status": "success",
                "detection_model": "ssd",
                "recognition_model": "facenet",
                "name": "didi_ruhyadi",
                "distance": 0.5,
                "dist_method": "cosine",
            }
        }
