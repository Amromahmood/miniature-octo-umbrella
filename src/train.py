from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def train():
    data = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data.data, data.target)
    output = MODEL_DIR / "iris_model.joblib"
    joblib.dump(model, output)
    print(f"Model saved to {output}")


if __name__ == "__main__":
    train()
