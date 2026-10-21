from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import main, predictor


client = TestClient(main.app)


class FakeModel:
    classes_ = np.array(["not_hate_speech", "hate_speech"])

    def __init__(self, predicted_class: str = "hate_speech") -> None:
        self.predicted_class = predicted_class

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.18, 0.82]]

    def predict(self, texts: list[str]) -> list[str]:
        return [self.predicted_class]


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_health_endpoint_reports_model_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "MODEL_PATH", predictor.Path("missing-model.joblib"))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_available"] is False


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_rejects_empty_or_whitespace_text(text: str) -> None:
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 422


def test_rejects_excessively_long_text() -> None:
    response = client.post("/predict", json={"text": "a" * 5001})
    assert response.status_code == 422


def test_hate_probability_is_selected_by_class_label() -> None:
    result = predictor.predict_with_model(FakeModel(), "sample text")
    assert result == {
        "is_hate": True,
        "result": "Hate Speech Detected",
        "hate_probability": 82.0,
        "message": "The model classified this text as hate speech.",
    }


def test_hate_and_non_hate_binary_outcomes() -> None:
    assert predictor.predict_with_model(FakeModel("hate_speech"), "text")["is_hate"] is True
    assert predictor.predict_with_model(FakeModel("not_hate_speech"), "text")["is_hate"] is False


def test_api_prediction_response_structure_and_probability_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "predict_text", lambda text: predictor.predict_with_model(FakeModel(), text))
    response = client.post("/predict", json={"text": "sample text"})
    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    assert set(payload) == {"is_hate", "result", "hate_probability", "message"}
    assert payload["is_hate"] is True
    assert 0 <= payload["hate_probability"] <= 100


def test_api_returns_non_hate_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "predict_text",
        lambda text: {
            **predictor.predict_with_model(FakeModel("not_hate_speech"), text),
            "hate_probability": 4.2,
        },
    )
    response = client.post("/predict", json={"text": "kind message"})
    assert response.status_code == 200
    assert response.json()["is_hate"] is False
    assert response.json()["result"] == "No Hate Speech Detected"
    assert response.json()["hate_probability"] == 4.2


def test_missing_model_is_reported_without_crashing_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(predictor, "MODEL_PATH", tmp_path / "missing.joblib")
    predictor.load_model.cache_clear()
    try:
        response = client.post("/predict", json={"text": "sample text"})
        assert response.status_code == 503
        assert "train" in response.json()["detail"].lower()
        assert client.get("/health").json()["status"] == "ok"
    finally:
        predictor.load_model.cache_clear()