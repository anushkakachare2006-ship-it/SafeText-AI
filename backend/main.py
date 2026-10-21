"""FastAPI application for SafeText AI."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.predictor import (
    MODEL_PATH,
    ModelConfigurationError,
    ModelUnavailableError,
    predict_text,
)


app = FastAPI(title="SafeText AI", version="1.0.0")
local_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
configured_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys([*local_origins, *configured_origins])),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    is_hate: bool
    result: str
    hate_probability: float = Field(ge=0, le=100)
    message: str


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "SafeText AI backend is running."}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_available": MODEL_PATH.is_file(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, bool | float | str]:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be blank.")

    try:
        return predict_text(request.text)
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ModelConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="The text could not be analyzed.") from error