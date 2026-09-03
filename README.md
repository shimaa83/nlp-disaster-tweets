# Disaster Tweet Classification API

A production-ready NLP microservice that classifies tweets into **disaster-related** or **non-disaster** events.

The project demonstrates the difference between **building an ML model** and **building a production-ready ML system** by combining machine learning, API engineering, testing, observability, model serialization, and containerization.

---

## Overview

The system uses a Scikit-Learn machine learning pipeline consisting of:

* Text preprocessing
* TF-IDF feature extraction
* Logistic Regression classification
* Pickle serialization for reproducible Python-based inference
* ONNX export for portable production inference
* ONNX Runtime
* FastAPI REST API
* Pydantic request/response validation
* Structured JSON logging
* Correlation IDs for request tracing
* Automated testing with pytest
* Pre-commit hooks with Ruff and Black
* Multi-stage Docker deployment
* Reproducible dependency management with `uv`

---

## Quickstart — From Zero to Prediction

### Prerequisites

* Python 3.11+
* [uv](https://docs.astral.sh/uv/)
* Git

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/shimaa83/nlp-disaster-tweets.git
cd nlp-disaster-tweets
uv sync
```

### Train the Model

The trained model artifacts are intentionally excluded from Git because model files are generated artifacts.

Train the Scikit-Learn pipeline:

```bash
uv run python -m nlp_disaster_tweets.train
```

This creates:

```text
models/model.pkl
```

### Export to ONNX

Export the trained model to ONNX:

```bash
uv run python -m nlp_disaster_tweets.export
```

This creates:

```text
models/model.onnx
```

### Start the API

```bash
uv run uvicorn nlp_disaster_tweets.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## Example Prediction

Send a POST request to `/predict`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Severe wildfire spreading near residential area, immediate evacuation ordered!\"}"
```

Example response:

```json
{
  "prediction": 1,
  "model_version": "0.1.0",
  "correlation_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
  "latency_ms": 4.12
}
```

Where:

* `prediction = 1` → disaster-related
* `prediction = 0` → non-disaster
* `model_version` → version of the deployed model
* `correlation_id` → unique ID used to trace the request through logs
* `latency_ms` → prediction latency in milliseconds

---

## API Endpoints

| Endpoint         | Method | Description                        |
| ---------------- | ------ | ---------------------------------- |
| `/health`        | GET    | Check service and model health     |
| `/metadata`      | GET    | Return model and pipeline metadata |
| `/predict`       | POST   | Classify a single tweet            |
| `/predict/batch` | POST   | Classify multiple tweets           |

### Health Check

```bash
curl http://localhost:8000/health
```

### Metadata

```bash
curl http://localhost:8000/metadata
```

### Interactive Documentation

FastAPI automatically provides Swagger UI:

```text
http://localhost:8000/docs
```

---

## Machine Learning Pipeline

The training pipeline consists of three main stages:

```text
Raw Tweet
   ↓
Text Preprocessing
   ↓
TF-IDF
   ↓
Logistic Regression
   ↓
Prediction
```

### Text Preprocessing

Tweets are processed using:

* URL removal
* Emoji removal
* Lowercasing
* Punctuation removal
* Whitespace normalization
* Tokenization
* Stop-word removal
* Porter stemming

### Feature Extraction

TF-IDF is used to transform the cleaned tweet text into numerical features.

### Classifier

The classification model is:

```text
Logistic Regression
```

The complete Scikit-Learn pipeline is persisted as a Pickle artifact for reproducible Python-based training and inference.

---

## Model Serialization

The trained model is available in two serialization formats:

| Format | Runtime               | Purpose                             |
| ------ | --------------------- | ----------------------------------- |
| Pickle | Python / Scikit-Learn | Training and Python-based inference |
| ONNX   | ONNX Runtime          | Portable production inference       |

The ONNX artifact contains the **TF-IDF and Logistic Regression inference pipeline**.

Text preprocessing is performed separately before ONNX Runtime inference because the custom `TextPreprocessor` is not exported as part of the ONNX graph.

### Pickle Security

> **Security note:** Pickle files should only be loaded from trusted sources. Deserializing an untrusted Pickle file can execute arbitrary code.

---

## Pickle vs ONNX Performance

A local inference benchmark was performed to compare the two serialization formats.

| Metric                 |    Pickle |      ONNX |
| ---------------------- | --------: | --------: |
| Mean inference latency | 1.6048 ms | 0.1184 ms |
| Model size             | 371.20 KB | 394.55 KB |

ONNX achieved an observed **92.62% lower mean inference latency** in this benchmark.

The ONNX model is slightly larger than the Pickle artifact:

```text
Pickle: 371.20 KB
ONNX:   394.55 KB
```

The benchmark measures inference latency only and does not include model loading time.

---

## Serialization Parity

A dedicated automated test verifies that the Pickle and ONNX versions produce identical predictions for the same input samples.

The test compares predictions generated by:

```text
Scikit-Learn / Pickle
        vs
ONNX Runtime
```

The parity test passed successfully.

This helps ensure that exporting the model to ONNX does not change the classification behavior.

---

## Production API Design

The API loads the ONNX model **once during application startup** using FastAPI's lifespan mechanism.

```text
Application Startup
        ↓
Load ONNX Model
        ↓
Keep Model in Memory
        ↓
Receive Request
        ↓
Validate Input
        ↓
Preprocess Text
        ↓
Run Prediction
        ↓
Return Response
```

This avoids loading the model for every request and reduces unnecessary inference overhead.

### Request Validation

API requests are validated using Pydantic schemas before reaching the prediction layer.

### Correlation IDs

Each request receives a unique correlation ID.

The correlation ID is included in structured JSON logs and API responses, making it easier to trace requests and investigate failures.

---

## Observability

The service uses structured JSON logging instead of unstructured `print()` statements.

Logs include information useful for production debugging, including:

* Request lifecycle
* Correlation ID
* Model loading
* Prediction processing
* Validation errors
* Unexpected application errors

This provides a foundation for centralized log collection and request tracing.

---

## Project Structure

```text
nlp-disaster-tweets/
│
├── docker/
│   ├── Dockerfile
│   └── .dockerignore
│
├── reports/
│   └── module-1.md
│
├── src/
│   └── nlp_disaster_tweets/
│       ├── api/
│       │   ├── main.py
│       │   └── schemas.py
│       │
│       ├── config.py
│       ├── context.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       └── train.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_predict.py
│   ├── test_serialization.py
│   └── test_train.py
│
├── notebooks/
│   └── twitter-disaster-classification.ipynb
│
├── pyproject.toml
├── README.md
└── uv.lock
```

### Generated Artifacts

The following directories contain locally generated artifacts and are excluded from Git:

```text
data/
models/*.pkl
models/*.onnx
```

The training dataset and trained model artifacts are therefore generated locally rather than committed to the repository.

---

## Testing

Run the complete test suite:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

The project enforces a minimum coverage threshold of:

```text
70%
```

The current test suite contains:

```text
12 tests
```

All tests pass successfully.

### Test Coverage Areas

The automated tests cover:

* Text preprocessing
* Model training
* Predictor behavior
* API endpoints
* Request validation
* Error handling
* Serialization
* Pickle vs ONNX prediction parity

---

## Code Quality

Pre-commit hooks are configured to automatically check and format the code.

Configured tools include:

* Ruff
* Black
* End-of-file fixer
* Trailing whitespace fixer

Run all hooks manually:

```bash
uv run pre-commit run --all-files
```

All configured pre-commit hooks pass successfully.

---

## Docker

The application is packaged using a **multi-stage Docker build**.

### Build the Image

```bash
docker build -f docker/Dockerfile -t nlp-disaster-tweets:v0.1.0 .
```

### Run the Container

```bash
docker run -p 8000:8000 nlp-disaster-tweets:v0.1.0
```

The production container:

* Uses a multi-stage build
* Uses Python 3.11
* Runs as a non-root user
* Includes the trained ONNX model
* Includes required NLTK data
* Provides a Docker health check
* Exposes port `8000`

### Docker Hub

The production image is published under:

```text
shimaa83/nlp-disaster-tweets:v0.1.0
```

Pull the image with:

```bash
docker pull shimaa83/nlp-disaster-tweets:v0.1.0
```

Run it with:

```bash
docker run -p 8000:8000 shimaa83/nlp-disaster-tweets:v0.1.0
```

---

## MLOps Practices

This project applies several practical MLOps principles:

* Clean separation between data, features, training, prediction, and API layers
* Centralized configuration
* Model versioning
* Startup model loading
* Pydantic input/output validation
* Structured JSON logging
* Correlation ID tracking
* Unit and API testing
* Coverage enforcement
* Serialization parity testing
* Pre-commit code quality checks
* ONNX-based production inference
* Multi-stage Docker builds
* Non-root container execution
* Reproducible dependency management with `uv`
* Docker Hub image publishing

---

## Module 1 Deliverables

The Module 1 implementation includes:

* Python package
* FastAPI service
* Four API endpoints
* Pydantic validation
* Startup model loading
* Structured JSON logs
* Correlation IDs
* Pickle and ONNX serialization
* ONNX parity testing
* Pytest test suite
* Coverage gate ≥ 70%
* Multi-stage Docker image
* Non-root container execution
* Docker Hub deployment
* Module 1 technical report
* Git-based development workflow

The release tag `v0.1.0` can be created after final Docker Hub verification.

---

## Development Workflow

Typical development workflow:

```text
Develop
   ↓
Run Tests
   ↓
Run Coverage
   ↓
Run Pre-commit
   ↓
Train / Export Model
   ↓
Benchmark
   ↓
Build Docker Image
   ↓
Test Container
   ↓
Publish Docker Image
   ↓
Release
```

Useful commands:

```bash
uv run pytest
```

```bash
uv run pre-commit run --all-files
```

```bash
uv run python -m nlp_disaster_tweets.train
```

```bash
uv run python -m nlp_disaster_tweets.export
```

```bash
docker build -f docker/Dockerfile -t nlp-disaster-tweets:v0.1.0 .
```

---

## Module 1 Report

A detailed technical report is available at:

```text
reports/module-1.md
```

The report documents:

* System architecture
* Machine learning pipeline
* API implementation
* Serialization strategy
* Pickle vs ONNX performance
* Model size comparison
* Serialization parity
* Testing and coverage
* Observability
* Dockerization
* MLOps maturity assessment
* Module 1 implementation status

---

## Future Improvements

Potential improvements for future modules include:

* CI/CD pipeline
* Automated Docker image builds
* Model registry
* Experiment tracking
* Automated model evaluation
* Monitoring and drift detection
* Automated deployment
* Load testing and performance optimization
* Model retraining workflows

---

## License

This project is developed for educational and MLOps practice purposes.
