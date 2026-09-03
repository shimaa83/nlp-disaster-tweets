import logging
from pathlib import Path
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

from .config import get_settings
from .predict import DisasterTweetPredictor

logger = logging.getLogger(__name__)


def export_to_onnx(model_path: Path, onnx_output_path: Path) -> None:
    """Export trained Scikit-Learn TF-IDF + Classifier steps to ONNX format."""
    if not model_path.exists():
        logger.error(f"Model load failure: Pickle model not found at {model_path}")
        raise FileNotFoundError(f"Pickle model not found at {model_path}")

    predictor = DisasterTweetPredictor.load(model_path)
    full_pipeline = predictor.model

    # استخراج خطوة الـ Preprocessor وبقية الـ Pipeline
    tfidf_step = full_pipeline.named_steps["tfidf"]
    classifier_step = full_pipeline.named_steps["classifier"]

    # إنشاء Pipeline فرعي قابل للتحويل لـ ONNX (بدون الـ Custom TextPreprocessor)
    onnx_pipeline = Pipeline(
        steps=[
            ("tfidf", tfidf_step),
            ("classifier", classifier_step),
        ]
    )

    # المدخل هنا عبارة عن السلاسل النصية المنظفة المنفردة
    initial_type = [("text_input", StringTensorType([None, 1]))]

    logger.info("Converting TF-IDF + Classifier pipeline to ONNX format...")
    onnx_model = to_onnx(onnx_pipeline, initial_types=initial_type)

    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"ONNX model successfully saved to {onnx_output_path}")


def main() -> None:
    settings = get_settings()
    export_to_onnx(settings.model_path, settings.onnx_path)


if __name__ == "__main__":
    main()
