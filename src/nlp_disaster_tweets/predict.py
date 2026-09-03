import logging
from pathlib import Path
import numpy as np
import onnxruntime as ort

from .features import TextPreprocessor

logger = logging.getLogger(__name__)


class DisasterTweetPredictor:
    """Predictor class using ONNX runtime and TextPreprocessor."""

    def __init__(self, onnx_path: Path):
        if not onnx_path.exists():
            logger.error(f"ONNX model not found at {onnx_path}")
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name
        self.preprocessor = TextPreprocessor()
        logger.info(f"Loaded ONNX model successfully from {onnx_path}")

    @classmethod
    def load(cls, onnx_path: Path) -> "DisasterTweetPredictor":
        return cls(onnx_path)

    def predict_one(self, sample: dict) -> int:
        """Process input text and run inference via ONNX Runtime."""
        raw_text = sample.get("text", "")

        # 1. تطبيق المعالجة النصية (Text Cleaning / Stemming)
        cleaned_text = self.preprocessor.transform([raw_text])[0]

        # 2. تجهيز الإدخال كـ 2D numpy array بصيغة string
        input_data = np.array([[cleaned_text]], dtype=object)

        # 3. تشغيل الاستنتاج عبر ONNX Runtime
        raw_pred = self.session.run(None, {self.input_name: input_data})[0][0]

        return int(raw_pred)
