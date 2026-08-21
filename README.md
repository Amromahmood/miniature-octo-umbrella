# Mini MLOps Pipeline

A beginner-friendly MLOps project that demonstrates a complete machine-learning workflow: data preparation, model training, model artifact creation, API serving, testing, Docker packaging, and CI.

## Project structure

```text
.
├── src/
│   ├── train.py
│   └── api.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml
```

## Run locally

```bash
python -m pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload
```

Open the API documentation at `http://127.0.0.1:8000/docs`.

Example request:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

## Docker

```bash
docker build -t mini-mlops .
docker run -p 8000:8000 mini-mlops
```

## What this project demonstrates

- Reproducible model training
- Model serialization with joblib
- REST inference API with FastAPI
- Automated tests with pytest
- Containerization with Docker
- Continuous integration with GitHub Actions

This is an educational starter project and is intentionally small so each MLOps component can be understood and extended.