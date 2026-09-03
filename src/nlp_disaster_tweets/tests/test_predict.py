from src.nlp_disaster_tweets.predict import DisasterTweetPredictor


def test_predict_returns_valid_range_and_deterministic(
    trained_model: DisasterTweetPredictor,
    sample_features: dict[str, str],
):
    """Test prediction returns correct binary range and is deterministic across calls."""
    pred1 = trained_model.predict_one(sample_features)
    pred2 = trained_model.predict_one(sample_features)

    # التحقق من أن النتيجة 0 أو 1
    assert isinstance(pred1, int)
    assert pred1 in (0, 1)

    # التحقق من الحتمية (Deterministic output)
    assert pred1 == pred2
