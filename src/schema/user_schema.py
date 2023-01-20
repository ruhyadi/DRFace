"""API User Schema."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from bson import ObjectId
from pydantic import BaseModel, Field

from src.schema.common_schema import PyObjectId


class UserSchema(BaseModel):
    """Base user schema."""

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "id": "63b3e159ea836a6d304ca105",
                "username": "didiruhyadi",
                "password": "satuduatiga",
            }
        }
