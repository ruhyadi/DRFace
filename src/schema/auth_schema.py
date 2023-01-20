"""API Authentication schema."""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from pydantic import BaseModel, Field


class CurrentUser(BaseModel):
    """Current user schema for JWT."""

    username: str = Field(..., description="Username")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        schema_extra = {
            "example": {
                "username": "didiruhyadi",
            }
        }


class Token(BaseModel):
    """Token schema."""

    access_token: str = Field(..., description="Access token")
    token_type: str = Field(..., description="Token type")

    class Config:
        schema_extra = {
            "example": {
                "access_token": "fake-satuduatiga",
                "token_type": "bearer",
            }
        }
