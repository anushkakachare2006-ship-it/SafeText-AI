"""Load the trained pipeline and return binary text predictions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "hate_speech_pipeline.joblib"
HATE_CLASS = "hate_speech"


class ModelUnavailableError(RuntimeError):
    """Raised when a trained model artifact is not available."""


class ModelConfigurationError(RuntimeError):
    """Raised when the model does not satisfy the prediction contract."""


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.is_file():
        raise ModelUnavailableError(
            "The model has not been trained. Run `python backend/train_model.py` first."
        )

    model = joblib.load(MODEL_PATH)
    classes = getattr(model, "classes_", None)
    if classes is None or HATE_CLASS not in classes:
        raise ModelConfigurationError(
            f"The trained model must include the {HATE_CLASS!r} class."
        )
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelConfigurationError("The trained model must support predict_proba().")
    return model


def predict_with_model(model: Any, text: str) -> dict[str, bool | float | str]:
    classes = list(getattr(model, "classes_", []))
    if HATE_CLASS not in classes:
        raise ModelConfigurationError(
            f"The trained model must include the {HATE_CLASS!r} class."
        )

    probabilities = model.predict_proba([text])[0]
    hate_probability = float(probabilities[classes.index(HATE_CLASS)])
    if not np.isfinite(hate_probability) or not 0 <= hate_probability <= 1:
        raise ModelConfigurationError("The model returned an invalid probability.")

    predicted_class = str(model.predict([text])[0])
    is_hate = predicted_class == HATE_CLASS
    result = "Hate Speech Detected" if is_hate else "No Hate Speech Detected"
    message = (
        "The model classified this text as hate speech."
        if is_hate
        else "The model did not classify this text as hate speech."
    )
    return {
        "is_hate": is_hate,
        "result": result,
        "hate_probability": round(hate_probability * 100, 1),
        "message": message,
    }


def predict_text(text: str) -> dict[str, bool | float | str]:
    return predict_with_model(load_model(), text)