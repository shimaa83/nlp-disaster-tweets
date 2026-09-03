import pytest
from src.nlp_disaster_tweets.features import TextPreprocessor


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


@pytest.mark.parametrize(
    "input_text, expected_tokens",
    [
        # حالة النص العادي مع إزالة Stopwords و URLs
        ("Check out http://example.com for fire news!", ["check", "fir", "new"]),
        # حالة الحروف الخاصة والرموز التعبيرية Emojis
        ("Disaster fire 🔥 near city!!!", ["disast", "fir", "near", "citi"]),
        # حالة النص المكون من مسافات ورموز فقط (Edge Case)
        ("   !!!  ---   ", [""]),
        # حالة الأحرف الكبيرة وتحويلها لكلمات مشتقة (Stemming)
        ("RUNNING FLOODS IN THE STREETS", ["run", "flood", "street"]),
    ],
)
def test_text_preprocessor_edge_cases(
    preprocessor: TextPreprocessor,
    input_text: str,
    expected_tokens: list[str],
):
    """Test feature preprocessor against various text edge cases."""
    transformed = preprocessor.transform([input_text])
    assert isinstance(transformed, list)
    assert len(transformed) == 1

    for token in expected_tokens:
        if token:
            assert token in transformed[0]
