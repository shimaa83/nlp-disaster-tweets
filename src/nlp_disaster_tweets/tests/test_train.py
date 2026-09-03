from pathlib import Path

import pandas as pd

from nlp_disaster_tweets import train


def test_build_pipeline():
    pipeline = train.build_pipeline()

    assert "preprocessor" in pipeline.named_steps
    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_train_model(monkeypatch, tmp_path):
    sample_df = pd.DataFrame(
        {
            "text": [
                "forest fire in the city",
                "everything is safe today",
                "massive earthquake reported",
                "beautiful sunny weather",
            ],
            "target": [1, 0, 1, 0],
        }
    )

    monkeypatch.setattr(train, "load_data", lambda _: sample_df)
    monkeypatch.setattr(train, "prepare_data", lambda df: df)

    monkeypatch.setattr(
        train,
        "split_data",
        lambda df, test_size, random_state: (
            df["text"].iloc[:2],
            df["text"].iloc[2:],
            df["target"].iloc[:2],
            df["target"].iloc[2:],
        ),
    )

    model_path = tmp_path / "model.pkl"

    model = train.train_model(
        data_path=Path("fake.csv"),
        model_path=model_path,
    )

    assert model_path.exists()
    assert "classifier" in model.named_steps


def test_main(monkeypatch):
    called = {}

    def fake_train_model(data_path, model_path):
        called["data_path"] = data_path
        called["model_path"] = model_path

    monkeypatch.setattr(train, "train_model", fake_train_model)

    train.main()

    assert "data_path" in called
    assert "model_path" in called


def test_train_module_execution(monkeypatch):
    import runpy

    monkeypatch.setattr(train, "main", lambda: None)

    runpy.run_module(
        "nlp_disaster_tweets.train",
        run_name="__main__",
    )
