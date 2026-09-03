from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from nlp_disaster_tweets.api.main import app, ml_models


def test_health_endpoint_healthy(client: TestClient):
    """Test /health returns 200 when model is present."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predict_happy_path_with_mock(monkeypatch):
    """Test /predict using a mock so test does not depend on real training run."""
    # إنشاء كائن Mock للنموذج
    mock_predictor = MagicMock()
    mock_predictor.predict_one.return_value = 1
    ml_models["predictor"] = mock_predictor

    with TestClient(app) as test_client:
        payload = {"text": "Emergency evacuation order issued"}
        response = test_client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1
        assert "latency_ms" in data
        assert "correlation_id" in data

    ml_models.clear()


def test_predict_invalid_payload_returns_422(client: TestClient):
    """Test /predict returns 422 for invalid payloads."""
    # إرسال نص فارغ ينتهك قيد min_length=1
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "details" in response.json()
