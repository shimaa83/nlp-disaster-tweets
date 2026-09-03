# Disaster Tweet Classification API

A production-ready NLP microservice that classifies tweets into **disaster-related** or **non-disaster** events.

The project uses a Scikit-Learn pipeline with:

* Text preprocessing
* TF-IDF feature extraction
* Logistic Regression classification
* Pickle serialization for the trained model
* ONNX export for portable inference
* ONNX Runtime for production prediction
* FastAPI REST API
* Pydantic request/response validation
* Structured JSON logging
* Correlation IDs for request tracing
* Automated tests with pytest
* Pre-commit hooks with Ruff and Black
* Dockerized deployment

The project demonstrates the difference between **building an ML model** and **building a production-ready ML system**.

---

## Quickstart — Zero to Prediction

### Prerequisites

* Python 3.11+
* [uv](https://docs.astral.sh/uv/)
* Git

### 3 Commands

```bash
# 1. Clone the repository and install dependencies
git clone https://github.com/shimaa83/nlp-disaster-tweets.git
cd nlp-disaster-tweets
uv sync

# 2. Export the trained model to ONNX
uv run python -m nlp_disaster_tweets.export

# 3. Start the API
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
* `latency_ms` → prediction latency

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

---

## Machine Learning Pipeline

The training pipeline consists of three stages:

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

Tweets are cleaned using:

* URL removal
* Emoji removal
* Lowercasing
* Punctuation removal
* Whitespace normalization
* Tokenization
* Stop-word removal
* Porter stemming

### Model

The classifier is:

```text
Logistic Regression
```

with TF-IDF features.

The complete Scikit-Learn pipeline is saved as a Pickle artifact for reproducible training/inference.

---

## Model Serialization

The trained model is available in two serialization formats:

| Format | Runtime               | Purpose                             |
| ------ | --------------------- | ----------------------------------- |
| Pickle | Python / Scikit-Learn | Training and Python-based inference |
| ONNX   | ONNX Runtime          | Portable production inference       |

The ONNX model can be executed using **ONNX Runtime** without requiring the original Scikit-Learn runtime for inference.

A serialization parity test verifies that Pickle and ONNX produce equivalent predictions on the same inputs.

> **Security note:** Pickle files should only be loaded from trusted sources because deserializing an untrusted Pickle file can execute arbitrary code.

---

## Production API Design

The API loads the model **once during application startup** using FastAPI's lifespan mechanism.

```text
Application Startup
        ↓
Load Model
        ↓
Keep Model in Memory
        ↓
Concurrent API Requests
        ↓
Prediction
```

This avoids loading the model for every request and significantly reduces unnecessary latency.

Requests are validated using Pydantic schemas before reaching the prediction layer.

Each request receives a unique correlation ID that is included in structured JSON logs, making it easier to trace and debug requests end-to-end.

---

## Project Structure

```text
nlp-disaster-tweets/
│
├── data/
│   └── train.csv
│
├── docker/
│   ├── Dockerfile
│   └── .dockerignore
│
├── models/
│   ├── model.pkl
│   └── model.onnx
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
│       ├── base.py
│       ├── config.py
│       ├── context.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── models.py
│       ├── predict.py
│       └── train.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_predict.py
│   └── test_serialization.py
│
├── notebooks/
│   └── twitter-disaster-classification.ipynb
│
├── pyproject.toml
├── README.md
└── uv.lock
```

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

The test suite covers:

* Text preprocessing
* Predictor behavior
* API endpoints
* Request validation
* Serialization
* Pickle vs ONNX prediction parity

The project targets a minimum test coverage of **70%**.

---

## Code Quality

Pre-commit hooks are configured to automatically check and format the code before commits.

Tools include:

* Ruff
* Black
* End-of-file fixer
* Trailing whitespace fixer

Run all hooks manually:

```bash
uv run pre-commit run --all-files
```

---

## Docker

The application is containerized using a multi-stage Docker build.

Build the image:

```bash
docker build -f docker/Dockerfile -t nlp-disaster-tweets .
```

Run the container:

```bash
docker run -p 8000:8000 nlp-disaster-tweets
```

The production container runs the application as a **non-root user**.

---

## MLOps Practices

This project applies several MLOps practices:

* Clean separation between data, features, training, prediction, and API layers
* Centralized configuration
* Model versioning
* Startup model loading
* Pydantic input/output validation
* Structured JSON logging
* Correlation ID tracking
* Unit and API testing
* Serialization parity testing
* Pre-commit code quality checks
* ONNX-based production inference
* Docker containerization
* Reproducible dependency management with `uv`

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
* Pickle and ONNX model artifacts
* ONNX parity testing
* Pytest test suite
* Coverage gate ≥ 70%
* Multi-stage Docker image
* Non-root container execution
* Docker Hub deployment
* Module 1 report
* GitHub Pull Request workflow
* `v0.1.0` release tag

---

## License

This project is developed by Me for educational and MLOps practice purposes.
