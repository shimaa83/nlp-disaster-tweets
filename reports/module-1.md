# Module 1 — Packaging, API, Serialization & Containerization Report

## 1. Executive Summary

This module focuses on transforming the Disaster Tweet Classification model from a development artifact into a production-ready machine learning service.

The implemented system includes:

* A pip-installable Python package
* Clean separation of data, features, training, prediction, and API layers
* FastAPI REST API
* Pydantic input/output validation
* Startup model loading
* Structured JSON logging
* Correlation IDs
* Pickle and ONNX model serialization
* Pickle/ONNX prediction parity testing
* Automated pytest testing
* Coverage enforcement at 70%
* Multi-stage Docker image
* Non-root container execution
* Docker health check
* Reproducible dependency management using `uv`

---

# 2. System Architecture

The production inference flow is:

```text
Client
  |
  v
FastAPI
  |
  v
Pydantic Validation
  |
  v
Correlation ID Middleware
  |
  v
Text Preprocessing
  |
  v
ONNX Runtime
  |
  v
Prediction
  |
  v
Structured JSON Response
```

The model is loaded once during application startup and kept in memory to avoid loading the model for every request.

---

# 3. Machine Learning Pipeline

The trained model consists of:

```text
Raw Tweet
    |
    v
Text Preprocessing
    |
    v
TF-IDF Vectorization
    |
    v
Logistic Regression
    |
    v
Binary Classification
```

The text preprocessing stage includes cleaning and normalization operations before TF-IDF feature extraction.

---

# 4. API Implementation

The FastAPI service provides four main endpoints:

| Endpoint         | Method | Purpose                          |
| ---------------- | ------ | -------------------------------- |
| `/health`        | GET    | Service and model health         |
| `/metadata`      | GET    | Model metadata and artifact hash |
| `/predict`       | POST   | Single tweet prediction          |
| `/predict/batch` | POST   | Batch tweet predictions          |

## Health Check

The production Docker container successfully loads the ONNX model into memory.

The health endpoint returned:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

This confirms successful integration between:

* Docker
* FastAPI
* ONNX Runtime
* ONNX model artifact

---

# 5. Example Prediction

A production API test was performed using:

```text
A massive wildfire is spreading rapidly and people are being evacuated
```

The API returned:

```json
{
  "prediction": 1,
  "model_version": "0.1.0",
  "correlation_id": "fb39375d-00d9-4884-baf2-9ed086e2973a",
  "latency_ms": 0.8
}
```

The request was successfully processed by the ONNX Runtime model running inside the Docker container.

---

# 6. Model Serialization

Two serialization formats were implemented:

| Format | Runtime      | Purpose                                  |
| ------ | ------------ | ---------------------------------------- |
| Pickle | Scikit-Learn | Model storage and Python-based inference |
| ONNX   | ONNX Runtime | Portable production inference            |

The ONNX model is used by the production FastAPI service.

---

# 7. Pickle vs ONNX Parity

A serialization parity test was implemented to verify that Pickle and ONNX produce identical predictions for the same input samples.

The test successfully passed:

```text
1 passed
```

The full test suite also passed with coverage above the required threshold.

This provides confidence that exporting the model to ONNX did not change the prediction behavior.

---

# 8. Inference Latency Benchmark

A benchmark was performed using the same dataset samples for both serialization formats.

The measured results were:

| Metric              | Pickle / Scikit-Learn | ONNX Runtime |
| ------------------- | --------------------: | -----------: |
| Mean latency        |             1.6048 ms |    0.1184 ms |
| Latency improvement |                     — |       92.62% |
| Model size          |             371.20 KB |    394.55 KB |

## Latency Analysis

ONNX Runtime achieved a mean inference latency of:

```text
0.1184 ms
```

compared with:

```text
1.6048 ms
```

for the Pickle/Scikit-Learn model.

This represents an observed latency reduction of:

```text
92.62%
```

under the benchmark conditions.

The result supports using ONNX Runtime as the production inference engine.

---

# 9. Model Size Comparison

The serialized model sizes were:

```text
Pickle : 371.20 KB
ONNX   : 394.55 KB
```

The ONNX artifact is approximately:

```text
23.35 KB
```

larger than the Pickle artifact.

Although ONNX is slightly larger, the increase is small compared with the measured inference latency improvement.

Therefore, the ONNX representation provides a favorable trade-off for production inference.

---

# 10. Testing

The project uses pytest with automated coverage enforcement.

The configured minimum coverage threshold is:

```text
70%
```

The current full test suite achieved:

```text
73%
```

The tests cover:

* Text preprocessing
* Predictor behavior
* API endpoints
* Request validation
* Model loading
* Serialization
* Pickle/ONNX parity

Testing includes fixtures and parametrized test cases.

---

# 11. Observability

The API implements structured JSON logging.

Example log information includes:

```text
timestamp
level
logger
message
correlation_id
```

Each HTTP request receives a correlation ID.

The ID is also returned through the:

```text
X-Request-ID
```

response header and included in prediction responses.

This enables request tracing across application logs.

---

# 12. Dockerization

The application is packaged using a multi-stage Docker build.

The Docker image:

* Uses Python 3.11 slim
* Uses a separate builder stage
* Installs production dependencies only in the runtime image
* Includes the trained ONNX model
* Includes required NLTK data
* Runs as a non-root user
* Exposes port 8000
* Includes a Docker health check
* Configures the required `en_US.UTF-8` locale for ONNX Runtime

The final container was successfully started and reported:

```text
Application startup complete.
Loaded ONNX model successfully.
GET /health HTTP/1.1" 200 OK
```

---

# 13. Production Model Packaging

The production Docker image contains the trained model artifact:

```text
/app/models/model.onnx
```

Therefore, users running the published Docker image do not need the training dataset to perform inference.

The dataset is required for training/retraining, but not for production prediction.

---

# 14. MLOps Practices Implemented

The following MLOps practices were implemented:

| Practice                     | Status    |
| ---------------------------- | --------- |
| Package structure            | Completed |
| Dependency management        | Completed |
| Configuration management     | Completed |
| Model serialization          | Completed |
| ONNX export                  | Completed |
| Serialization parity testing | Completed |
| FastAPI serving              | Completed |
| Input validation             | Completed |
| Startup model loading        | Completed |
| Structured logging           | Completed |
| Correlation IDs              | Completed |
| Automated testing            | Completed |
| Coverage enforcement         | Completed |
| Docker containerization      | Completed |
| Multi-stage Docker build     | Completed |
| Non-root execution           | Completed |
| Health checks                | Completed |

---

# 15. MLOps Maturity Self-Assessment

## Level 1 — Development

The original model was developed as a machine learning pipeline using Scikit-Learn.

## Level 2 — Packaging

The project was converted into a structured, pip-installable Python package with separated modules for data, features, training, prediction, and configuration.

## Level 3 — Serving

The model was exposed through a FastAPI service with validated inputs, structured outputs, health checks, metadata, and batch prediction.

## Level 4 — Reproducibility

Dependencies are managed using `uv`, and the project includes automated tests, coverage enforcement, and pre-commit quality checks.

## Level 5 — Production Containerization

The application is packaged as a multi-stage Docker image, runs as a non-root user, contains the production model artifact, and exposes a health endpoint.

### Current Assessment

The project has progressed from a standalone ML experiment to a **containerized production-oriented ML inference service**.

Future maturity improvements should focus on CI/CD, model registry, monitoring, drift detection, experiment tracking, and automated retraining.

---

# 16. Remaining Module 1 Release Tasks

The remaining release activities are:

* [ ] Publish Docker image to Docker Hub
* [ ] Complete peer review by two reviewers
* [ ] Merge `module-1-packaging` into `master`
* [ ] Create release tag `v0.1.0`

---

# 17. Conclusion

Module 1 successfully transforms the Disaster Tweet Classification model into a production-oriented machine learning service.

The key production improvement is the adoption of ONNX Runtime, which achieved a measured **92.62% lower mean inference latency** than the Pickle/Scikit-Learn implementation under the benchmark conditions.

The final system combines machine learning, API engineering, testing, observability, serialization, and containerization into a reproducible MLOps workflow.
