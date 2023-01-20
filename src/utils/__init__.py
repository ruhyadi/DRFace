"""Common utils for the DRFace."""

def exclude_fields(fields: list) -> dict:
    """Return a dict with the fields to be excluded."""
    return {field: {"exclude": True} for field in fields}