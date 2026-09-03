from typing import List
from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    """Schema for single tweet prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The tweet text to classify.",
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "Forest fire near la porte texas usa"}}
    )


class BatchPredictionRequest(BaseModel):
    """Schema for batch tweet prediction request."""

    inputs: List[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of prediction inputs (1 to 100).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": [
                    {"text": "Forest fire near la porte texas usa"},
                    {"text": "Just enjoying my coffee this morning"},
                ]
            }
        }
    )


class PredictionResponse(BaseModel):
    """Schema for single prediction response."""

    prediction: int = Field(..., description="Target class (0 or 1).")
    model_version: str = Field(..., description="Model version tag.")
    correlation_id: str = Field(..., description="Traceability request ID.")
    latency_ms: float = Field(..., description="Processing latency in milliseconds.")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""

    predictions: List[int] = Field(..., description="List of target classes (0 or 1).")
    model_version: str = Field(..., description="Model version tag.")
    correlation_id: str = Field(..., description="Traceability request ID.")
    latency_ms: float = Field(..., description="Processing latency in milliseconds.")
