# Mini MLOps Pipeline

A small, practical MLOps starter project built around a scikit-learn model. The goal is to show the pieces of a machine-learning workflow that are easy to miss when learning ML: reproducible training, model artifacts, an inference API, automated tests, containerization, and continuous integration.

> **Status:** early-stage educational project. The code is intentionally small and easy to extend.

## What it does

The project trains a Random Forest classifier on the Iris dataset and exposes the trained model through a FastAPI service.

The API provides:

- `GET /health` — service and model status
- `GET /model-info` — training metadata and evaluation information
- `POST /predict` — class prediction and confidence score

## Project structure

```text
.
├── .github/workflows/ci.yml
├── src/
│   ├── train.py
│   └── api.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Run locally

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

Example request:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

## Run tests

```bash
pytest
```

## Docker

```bash
docker build -t mini-mlops .
docker run -p 8000:8000 mini-mlops
```

## CI

Every push and pull request runs the GitHub Actions workflow. CI installs the dependencies, trains the model from scratch, and runs the test suite.

## Why this project exists

Many beginner ML projects stop after training a model in a notebook. This project focuses on the next step: turning that model into a small, reproducible service that can be tested and packaged.

## Roadmap

- [x] Reproducible model training
- [x] Model metadata and evaluation metric
- [x] FastAPI inference endpoint
- [x] Automated tests
- [x] Docker image
- [x] GitHub Actions CI
- [ ] Add structured logging
- [ ] Add model versioning
- [ ] Add a small data-validation stage
- [ ] Add an example deployment guide

## Contributing

Small improvements, bug reports, documentation fixes, and tests are welcome. See `CONTRIBUTING.md` for the basic workflow.

## License

MIT License. See `LICENSE`.
