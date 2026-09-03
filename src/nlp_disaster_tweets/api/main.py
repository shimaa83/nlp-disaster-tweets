import hashlib
import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..context import get_correlation_id, set_correlation_id
from ..logging_conf import setup_logging
from ..predict import DisasterTweetPredictor
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# كائن حفظ النموذج في الذاكرة لتجنب إعادة التحميل في كل طلب
ml_models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load the ONNX model ONCE at startup."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)

    logger.info(
        "Initializing API application. ONNX path: %s",
        settings.onnx_path,
    )

    try:
        logger.info(
            "Checking ONNX model path. Exists: %s",
            settings.onnx_path.exists(),
        )

        if not settings.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {settings.onnx_path}")

        predictor = DisasterTweetPredictor.load(settings.onnx_path)

        ml_models["predictor"] = predictor

        logger.info(
            "ONNX Model successfully loaded into memory from %s",
            settings.onnx_path,
        )

    except Exception:
        logger.exception(
            "Model load failure at startup. ONNX path: %s",
            settings.onnx_path,
        )
        ml_models["predictor"] = None

    yield

    ml_models.clear()
    logger.info("Cleaning up application resources...")


app = FastAPI(
    title="NLP Disaster Tweets Classification API",
    version=get_settings().model_version,
    lifespan=lifespan,
)


# =========================
# Middlewares
# =========================


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Inject correlation_id (uuid4) and return X-Request-ID header."""
    correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_correlation_id(correlation_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response


# =========================
# Exception Handlers
# =========================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert validation errors into a clean 422 with useful message."""
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Invalid input format or constraints failed.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors with 500 status without leaking tracebacks to client."""
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please contact system support.",
        },
    )


# =========================
# Endpoints
# =========================


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint: 200 ONLY if model is loaded in memory."""
    predictor = ml_models.get("predictor")
    if predictor is None:
        logger.error("Health check failed: Model not loaded in memory")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "reason": "Model is not loaded in memory"},
        )
    return {"status": "healthy", "model_loaded": True}


@app.get("/metadata", status_code=status.HTTP_200_OK)
def get_metadata():
    """Metadata endpoint: returns model metadata and details."""
    settings = get_settings()

    # حساب الـ Hash لملف الـ ONNX لتأكيد سلامته
    artifact_hash = "N/A"
    target_path = (
        settings.onnx_path if settings.onnx_path.exists() else settings.model_path
    )
    if target_path.exists():
        with open(target_path, "rb") as f:
            artifact_hash = hashlib.sha256(f.read()).hexdigest()

    return {
        "model_version": settings.model_version,
        "framework": "onnxruntime",
        "artifact_hash": artifact_hash,
        "features": ["text"],
        "pipeline_steps": ["preprocessor", "tfidf", "classifier"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(request: PredictionRequest):
    """Predict category for a single tweet input."""
    predictor = ml_models.get("predictor")
    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "Model is not available"},
        )

    t0 = time.perf_counter()
    logger.debug(f"Input text feature: {request.text}")

    prediction = predictor.predict_one({"text": request.text})
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(f"Prediction served with latency: {latency_ms:.3f} ms")

    return PredictionResponse(
        prediction=prediction,
        model_version=get_settings().model_version,
        correlation_id=get_correlation_id() or "unknown",
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """Predict categories for a batch of tweet inputs."""
    predictor = ml_models.get("predictor")
    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "Model is not available"},
        )

    t0 = time.perf_counter()
    features = [{"text": item.text} for item in request.inputs]

    # دعم التوقع على الـ Batch إما عبر الدالة المباشرة أو التكرار
    if hasattr(predictor, "predict_batch"):
        predictions = predictor.predict_batch(features)
    else:
        predictions = [predictor.predict_one(item) for item in features]

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"Batch prediction served for {len(features)} items with latency: {latency_ms:.3f} ms"
    )

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=get_settings().model_version,
        correlation_id=get_correlation_id() or "unknown",
        latency_ms=round(latency_ms, 2),
    )
