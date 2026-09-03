from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NLP_DISASTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================
    # Paths
    # =========================

    data_path: Path = Path("data/train.csv")  # تم التعديل إلى CSV

    model_path: Path = Path("models/model.pkl")

    onnx_path: Path = Path("models/model.onnx")

    metadata_path: Path = Path("models/metadata.json")

    # =========================
    # Model
    # =========================

    model_version: str = "0.1.0"

    # =========================
    # Data
    # =========================

    validation_size: float = 0.20

    random_state: int = 42

    # =========================
    # TF-IDF
    # =========================

    tfidf_max_features: int | None = None

    tfidf_ngram_min: int = 1

    tfidf_ngram_max: int = 1

    # =========================
    # Logistic Regression
    # =========================

    logistic_dual: bool = True

    logistic_solver: str = "liblinear"

    logistic_max_iter: int = 5000

    # =========================
    # API
    # =========================

    host: str = "0.0.0.0"

    port: int = 8000

    # =========================
    # Logging
    # =========================

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
