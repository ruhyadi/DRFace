"""Enums schema."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from enum import Enum


class DetectionModel(str, Enum):
    """Detection model."""
    ssd = "ssd"


class RecognitionModel(str, Enum):
    """Recognition model."""
    facenet = "facenet"
    arcface = "arcface"