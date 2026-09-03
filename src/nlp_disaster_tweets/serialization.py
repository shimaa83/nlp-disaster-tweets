import statistics
import time
from pathlib import Path

import joblib
import onnxruntime as ort
import numpy as np

from nlp_disaster_tweets.config import get_settings
from nlp_disaster_tweets.data import load_data, prepare_data
from nlp_disaster_tweets.features import TextPreprocessor


def benchmark_pickle(model, texts, iterations=100):
    latencies = []

    for text in texts:
        for _ in range(iterations):
            start = time.perf_counter()
            model.predict([text])
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

    return latencies


def benchmark_onnx(session, input_name, preprocessor, texts, iterations=100):
    latencies = []

    cleaned_texts = preprocessor.transform(texts)

    for text in cleaned_texts:
        input_data = np.array([[text]], dtype=object)

        for _ in range(iterations):
            start = time.perf_counter()
            session.run(None, {input_name: input_data})
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

    return latencies


def summarize(name, latencies):
    latencies_sorted = sorted(latencies)

    p95_index = int(len(latencies_sorted) * 0.95) - 1

    print(f"\n{name}")
    print("-" * 40)
    print(f"Requests : {len(latencies)}")
    print(f"Mean     : {statistics.mean(latencies):.4f} ms")
    print(f"Median   : {statistics.median(latencies):.4f} ms")
    print(f"P95      : {latencies_sorted[p95_index]:.4f} ms")
    print(f"Min      : {min(latencies):.4f} ms")
    print(f"Max      : {max(latencies):.4f} ms")


def main():
    settings = get_settings()

    print("=" * 60)
    print("Pickle vs ONNX Serialization Benchmark")
    print("=" * 60)

    # Load dataset
    df = prepare_data(load_data(settings.data_path))

    # استخدام عينة ثابتة حتى يكون benchmark سريع وعادل
    texts = df["text"].iloc[:50].tolist()

    print(f"\nBenchmark samples: {len(texts)}")
    print("Iterations per sample: 100")

    # --------------------------------------------------
    # Pickle
    # --------------------------------------------------

    print("\nLoading Pickle model...")
    pickle_model = joblib.load(settings.model_path)

    # Warm-up
    pickle_model.predict([texts[0]])

    pickle_latencies = benchmark_pickle(
        pickle_model,
        texts,
    )

    # --------------------------------------------------
    # ONNX
    # --------------------------------------------------

    print("Loading ONNX model...")
    session = ort.InferenceSession(str(settings.onnx_path))
    input_name = session.get_inputs()[0].name
    preprocessor = TextPreprocessor()

    # Warm-up
    cleaned = preprocessor.transform([texts[0]])
    session.run(
        None,
        {
            input_name: np.array([[cleaned[0]]], dtype=object),
        },
    )

    onnx_latencies = benchmark_onnx(
        session,
        input_name,
        preprocessor,
        texts,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    summarize("Pickle / Scikit-Learn", pickle_latencies)
    summarize("ONNX Runtime", onnx_latencies)

    pickle_mean = statistics.mean(pickle_latencies)
    onnx_mean = statistics.mean(onnx_latencies)

    improvement = ((pickle_mean - onnx_mean) / pickle_mean) * 100

    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Pickle mean : {pickle_mean:.4f} ms")
    print(f"ONNX mean   : {onnx_mean:.4f} ms")
    print(f"Difference  : {improvement:.2f}%")

    # --------------------------------------------------
    # Model sizes
    # --------------------------------------------------

    pickle_size = Path(settings.model_path).stat().st_size
    onnx_size = Path(settings.onnx_path).stat().st_size

    print("\nModel Sizes")
    print("-" * 40)
    print(f"Pickle: {pickle_size / 1024:.2f} KB")
    print(f"ONNX  : {onnx_size / 1024:.2f} KB")


if __name__ == "__main__":
    main()
