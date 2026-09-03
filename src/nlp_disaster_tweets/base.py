from abc import ABC, abstractmethod
from typing import Any, Union


class BasePredictor(ABC):
    """Abstract interface for model predictors."""

    @classmethod
    @abstractmethod
    def load(cls, model_path: str) -> "BasePredictor":
        """Load a trained predictor from an artifact."""

    @abstractmethod
    def predict_one(self, features: dict[str, Any]) -> Union[int, float]:
        """Generate a prediction for one input."""

    @abstractmethod
    def predict_batch(
        self,
        features: list[dict[str, Any]],
    ) -> list[Union[int, float]]:
        """Generate predictions for multiple inputs."""
