import joblib
import numpy as np
import pytest
import onnxruntime as ort

from nlp_disaster_tweets.config import get_settings
from nlp_disaster_tweets.data import load_data, prepare_data
from nlp_disaster_tweets.features import TextPreprocessor


def test_pickle_onnx_parity():
    """Assert parity between pickle model and ONNX runtime predictions."""
    settings = get_settings()

    if not settings.onnx_path.exists():
        pytest.skip("ONNX artifact not found. Export ONNX model first.")

    if not settings.model_path.exists():
        pytest.skip("Pickle model not found. Train the model first.")

    # Load and prepare sample data
    df = load_data(settings.data_path)
    df = prepare_data(df)
    sample_texts = df["text"].iloc[:50].tolist()

    # ---------------------------------------------------------
    # 1. Pickle predictions
    # ---------------------------------------------------------
    pickle_model = joblib.load(settings.model_path)
    pkl_preds = pickle_model.predict(sample_texts)

    # ---------------------------------------------------------
    # 2. ONNX predictions
    # ---------------------------------------------------------
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.transform(sample_texts)

    session = ort.InferenceSession(str(settings.onnx_path))
    input_name = session.get_inputs()[0].name

    onnx_preds = []

    for text in cleaned_texts:
        input_data = np.array([[text]], dtype=object)
        pred = session.run(None, {input_name: input_data})[0][0]
        onnx_preds.append(pred)

    # ---------------------------------------------------------
    # 3. Verify parity
    # ---------------------------------------------------------
    np.testing.assert_array_equal(
        pkl_preds,
        onnx_preds,
        err_msg="Pickle and ONNX predictions are not identical.",
    )
