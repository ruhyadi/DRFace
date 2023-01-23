"""Logging to database schema."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np
from bson import ObjectId
from fastapi import Form
from pydantic import BaseModel, Field

from src.schema.common_schema import PyObjectId


class LogSchema(BaseModel):
    """Database logging schema."""

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="User ID")
    request_id: str = Field(..., description="Request ID")
    timestamp: str = Field(..., description="Timestamp")
    request_type: str = Field(..., description="Request type")
    request_data: dict = Field(..., description="Request data")
    response_data: dict = Field(..., description="Response data")
    response_time: float = Field(..., description="Response time")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}