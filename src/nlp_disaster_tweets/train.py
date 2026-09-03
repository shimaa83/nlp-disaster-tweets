from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import get_settings
from .data import load_data, prepare_data, split_data
from .features import TextPreprocessor


def build_pipeline() -> Pipeline:
    """Build the complete text classification pipeline."""

    settings = get_settings()

    return Pipeline(
        steps=[
            ("preprocessor", TextPreprocessor()),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=settings.tfidf_max_features,
                    ngram_range=(
                        settings.tfidf_ngram_min,
                        settings.tfidf_ngram_max,
                    ),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    dual=settings.logistic_dual,
                    solver=settings.logistic_solver,
                    max_iter=settings.logistic_max_iter,
                    random_state=settings.random_state,
                ),
            ),
        ]
    )


def train_model(
    data_path: Path,
    model_path: Path,
) -> Pipeline:
    """Train the classifier and persist the complete pipeline."""

    settings = get_settings()

    df = load_data(data_path)
    df = prepare_data(df)

    X_train, X_val, y_train, y_val = split_data(
        df=df,
        test_size=settings.validation_size,
        random_state=settings.random_state,
    )

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    return pipeline


def main() -> None:
    """Train the production model."""

    settings = get_settings()

    train_model(
        data_path=settings.data_path,
        model_path=settings.model_path,
    )


if __name__ == "__main__":
    main()
