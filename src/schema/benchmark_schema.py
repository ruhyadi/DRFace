"""Face recognition benchmark schema."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from fastapi import File, Form, UploadFile
from pydantic import BaseModel

from src.schema.enums_schema import RecognitionModel


class FaceBenchmarkRequest(BaseModel):
    """Face recognition benchmark request schema."""

    images_zip: UploadFile = File(..., description="Images zip file")
    model: RecognitionModel = Form(..., description="Recognition model")

    class Config:
        schema_extra = {
            "example": {
                "images_zip": "my_face_benchmark.zip",
                "model": "facenet",
            }
        }

class FaceBenchmarkResponse(BaseModel):
    """Face recognition benchmark response schema."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float

    class Config:
        schema_extra = {
            "example": {
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "f1_score": 0.9,
            }
        }
