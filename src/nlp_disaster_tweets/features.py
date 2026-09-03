import re
from typing import Any

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocess disaster tweet text for machine learning."""

    def __init__(self) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def fit(
        self,
        X: Any,
        y: Any = None,
    ) -> "TextPreprocessor":
        """Fit the transformer."""

        return self

    def transform(self, X: Any) -> list[str]:
        """Transform raw tweets into cleaned text."""

        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize a single tweet."""

        if not isinstance(text, str):
            raise TypeError("Each text input must be a string.")

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove emojis
        text = re.sub(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            "",
            text,
        )

        # Lowercase
        text = text.lower()

        # Remove special characters
        text = re.sub(r"[^\w\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenization
        tokens = word_tokenize(text)

        # Stopword removal
        tokens = [token for token in tokens if token not in self.stop_words]

        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]

        return " ".join(tokens)
