from pathlib import Path
import json
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "iris_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

app = FastAPI(title="Mini MLOps API", version="1.1.0")


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)


model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata is not available")
    return metadata


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run python src/train.py first.")

    probabilities = model.predict_proba([request.features])[0]
    prediction = int(model.predict([request.features])[0])
    confidence = float(max(probabilities))
    label = metadata["classes"][prediction] if metadata else str(prediction)

    return {
        "prediction": prediction,
        "class": label,
        "confidence": round(confidence, 4),
    }
