from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict
import json
import shutil
from pathlib import Path
import tempfile
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config
from api.inference import inference

app = FastAPI(
    title="🎵 Music Genre Classifier API",
    description="REST API for music genre classification using MLP, CNN, and Random Forest models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


class PredictionResponse(BaseModel):
    predicted_genre: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str


class ModelInfo(BaseModel):
    name: str
    accuracy: float
    f1_score: float
    training_time: float


@app.on_event("startup")
async def startup_event():
    inference.load_models()


@app.get("/")
async def root():
    return {
        "message": "🎵 Music Genre Classifier API",
        "version": "1.0.0",
        "documentation": "/docs",
        "available_models": ["mlp", "cnn", "rf"],
        "endpoints": {
            "predict": "POST /predict/{model}",
            "models": "GET /models",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": inference.models_loaded
    }


@app.get("/models")
async def get_models():
    models_info = {}

    for model_name, reports_dir in [
        ("mlp", config.MLP_REPORTS_DIR),
        ("cnn", config.CNN_REPORTS_DIR),
        ("rf", config.RF_REPORTS_DIR)
    ]:
        metrics_path = reports_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            models_info[model_name] = {
                "name": model_name.upper(),
                "test_accuracy": metrics.get("test_accuracy", 0.0),
                "test_f1": metrics.get("test_f1", 0.0),
                "training_time": metrics.get("training_time", 0.0),
                "description": {
                    "mlp": "Multi-Layer Perceptron - Neural network trained on audio features",
                    "cnn": "Convolutional Neural Network - Deep learning model trained on mel-spectrograms",
                    "rf": "Random Forest - Classical ML model trained on audio features"
                }[model_name]
            }

    return models_info


@app.post("/predict/mlp", response_model=PredictionResponse)
async def predict_mlp(file: UploadFile = File(...)):
    return await _predict_with_model(file, "mlp")


@app.post("/predict/cnn", response_model=PredictionResponse)
async def predict_cnn(file: UploadFile = File(...)):
    return await _predict_with_model(file, "cnn")


@app.post("/predict/rf", response_model=PredictionResponse)
async def predict_rf(file: UploadFile = File(...)):
    return await _predict_with_model(file, "rf")


async def _predict_with_model(file: UploadFile, model_name: str):
    if not inference.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {allowed_extensions}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        if model_name == "mlp":
            result = inference.predict_mlp(tmp_path)
        elif model_name == "cnn":
            result = inference.predict_cnn(tmp_path)
        elif model_name == "rf":
            result = inference.predict_rf(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")

        result["model_used"] = model_name.upper()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
