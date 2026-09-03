import numpy as np
import pytest
import onnxruntime as ort

from src.nlp_disaster_tweets.config import get_settings
from src.nlp_disaster_tweets.data import load_data, prepare_data
from src.nlp_disaster_tweets.features import TextPreprocessor
from src.nlp_disaster_tweets.predict import DisasterTweetPredictor


def test_pickle_onnx_parity():
    """Assert parity between pickle model and ONNX runtime predictions."""
    settings = get_settings()
    if not settings.onnx_path.exists():
        pytest.skip("ONNX artifact not found. Export ONNX model first.")

    df = load_data(settings.data_path)
    df = prepare_data(df)
    sample_texts = df["text"].iloc[:50].tolist()

    # 1. توقعات Pickle
    predictor = DisasterTweetPredictor.load(settings.model_path)
    pkl_preds = [predictor.predict_one({"text": t}) for t in sample_texts]

    # 2. توقعات ONNX (مع تطبيق المعالجة النصية أولاً)
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.transform(sample_texts)

    session = ort.InferenceSession(str(settings.onnx_path))
    input_name = session.get_inputs()[0].name

    onnx_preds = []
    for text in cleaned_texts:
        input_data = np.array([[text]], dtype=object)
        pred = session.run(None, {input_name: input_data})[0][0]
        onnx_preds.append(pred)

    # 3. التأكد من التطابق التام
    np.testing.assert_allclose(pkl_preds, onnx_preds, atol=1e-4)
