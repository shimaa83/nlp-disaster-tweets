import logging
from pathlib import Path

import joblib
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

from .config import get_settings

logger = logging.getLogger(__name__)


def export_to_onnx(model_path: Path, onnx_output_path: Path) -> None:
    """Export the trained Scikit-Learn pipeline to ONNX format."""
    if not model_path.exists():
        logger.error("Pickle model not found at %s", model_path)
        raise FileNotFoundError(f"Pickle model not found at {model_path}")

    # Load the trained Scikit-Learn pipeline directly.
    full_pipeline = joblib.load(model_path)

    # The custom TextPreprocessor is not exported to ONNX.
    # Preprocessing is performed before ONNX Runtime inference.
    tfidf_step = full_pipeline.named_steps["tfidf"]
    classifier_step = full_pipeline.named_steps["classifier"]

    # Create an ONNX-compatible pipeline containing
    # only TF-IDF and the classifier.
    onnx_pipeline = Pipeline(
        steps=[
            ("tfidf", tfidf_step),
            ("classifier", classifier_step),
        ]
    )

    initial_type = [("text_input", StringTensorType([None, 1]))]

    logger.info("Converting TF-IDF + Classifier pipeline to ONNX format...")

    onnx_model = to_onnx(
        onnx_pipeline,
        initial_types=initial_type,
    )

    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(onnx_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info("ONNX model successfully saved to %s", onnx_output_path)


def main() -> None:
    """Export the production model to ONNX."""
    settings = get_settings()

    export_to_onnx(
        model_path=settings.model_path,
        onnx_output_path=settings.onnx_path,
    )


if __name__ == "__main__":
    main()
