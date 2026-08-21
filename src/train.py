from pathlib import Path
import json
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def train():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, MODEL_DIR / "iris_model.joblib")
    metadata = {
        "dataset": "Iris",
        "algorithm": "RandomForestClassifier",
        "random_state": 42,
        "test_accuracy": round(float(accuracy), 4),
        "features": list(data.feature_names),
        "classes": list(data.target_names),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Model saved. Test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    train()
