from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "iris_model.joblib"

app = FastAPI(title="Mini MLOps API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: list[float]


model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        return {"error": "Model not trained. Run python src/train.py first."}
    prediction = int(model.predict([request.features])[0])
    return {"prediction": prediction}
