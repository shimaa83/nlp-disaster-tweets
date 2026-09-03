import pytest
from fastapi.testclient import TestClient

from src.nlp_disaster_tweets.api.main import app, ml_models
from src.nlp_disaster_tweets.config import get_settings
from src.nlp_disaster_tweets.predict import DisasterTweetPredictor


@pytest.fixture(scope="session")
def sample_features() -> dict[str, str]:
    """Fixture for single sample text feature."""
    return {"text": "Forest fire near la porte texas usa"}


@pytest.fixture(scope="session")
def trained_model():
    """Session-scoped fixture to load trained predictor once for test suite."""
    settings = get_settings()
    if not settings.model_path.exists():
        pytest.skip("Model artifact pickle not found. Run train script first.")
    return DisasterTweetPredictor.load(settings.model_path)


@pytest.fixture
def client(trained_model) -> TestClient:
    """Fixture for FastAPI TestClient with loaded model."""
    ml_models["predictor"] = trained_model
    with TestClient(app) as test_client:
        yield test_client
    ml_models.clear()
