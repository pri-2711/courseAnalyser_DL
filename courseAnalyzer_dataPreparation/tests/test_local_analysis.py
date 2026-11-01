import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd

from src.data_loader import normalize_reviews, iter_input
from src.evaluation import EvaluationAccumulator
from src.metadata import analyze_courses
from src.scoring import add_rows, distribution, empty_counts
from src.assistant import retrieve_reviews
from src.sbert_model import ReferenceSBERT
import run_analysis


class FakeAnalyzer:
    embedded = 0

    def __init__(self, **kwargs):
        pass

    def embed(self, texts):
        type(self).embedded += len(texts)
        return np.array([[1, 0, 0] if "positive" in text else
                         [0, 1, 0] if "negative" in text else [0, 0, 1] for text in texts], dtype=np.float32)

    def classify_embeddings(self, vectors, uncertainty_margin=0.03):
        scores = vectors @ np.eye(3, dtype=np.float32)
        labels = np.array(["positive", "negative", "neutral"])[scores.argmax(axis=1)]
        margins = np.full(len(vectors), 0.6)
        return scores, labels, margins, margins < uncertainty_margin

    def classify(self, texts, uncertainty_margin=0.03):
        return self.classify_embeddings(self.embed(texts), uncertainty_margin)

    def save_model(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "local_model.marker").write_text("saved", encoding="utf-8")

    def save_reference_embeddings(self, path):
        np.save(path, np.ones((8, 3), dtype=np.float32))


class LocalAnalysisTests(unittest.TestCase):
    def test_missing_platform_is_separate_unknown(self):
        frame = normalize_reviews(pd.DataFrame({"review": ["good", "fine"], "platform": ["Udemy", ""]}))
        self.assertEqual(frame.platform.tolist(), ["udemy", "unknown"])
        self.assertTrue(frame.rating.isna().all())

    def test_jsonl_content_with_json_extension_streams_as_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            path.write_text('{"review":"one"}\n{"review":"two"}\n', encoding="utf-8")
            chunks = list(iter_input(path, chunksize=1))
            self.assertEqual([len(chunk) for chunk in chunks], [1, 1])

    def test_distribution_and_confusion_matrix(self):
        counts = empty_counts()
        add_rows(counts, np.array(["positive", "negative", "neutral"]), np.array([False, True, False]))
        self.assertEqual(distribution(counts)["sentiment_value"], 0.0)
        accumulator = EvaluationAccumulator()
        accumulator.update(["positive", "negative", "neutral"], ["positive", "neutral", "neutral"])
        self.assertAlmostEqual(accumulator.report()["accuracy"], 2 / 3)
        self.assertEqual(accumulator.report()["confusion_matrix"][1][2], 1)

    def test_course_analytics_does_not_require_review_join(self):
        summary, normalized = analyze_courses(pd.DataFrame({
            "title": ["A", "B"], "platform": ["Udemy", ""], "rating": [4.5, 4.0],
            "price": [10, 20], "num_subscribers": [100, 50], "num_reviews": [20, 5]}))
        self.assertEqual(summary["course_count"], 2)
        self.assertEqual(summary["platform_distribution"]["unknown"], 1)
        self.assertEqual(normalized.iloc[0].subscribers, 100)

    def test_sbert_uses_persisted_model_and_reference_vectors_locally(self):
        class LocalOnlyStub:
            def __init__(self, path, **kwargs):
                self.path, self.kwargs = path, kwargs
            def encode(self, *args, **kwargs):
                raise AssertionError("Reference vectors should be loaded from disk")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model_path = root / "model"; model_path.mkdir()
            refs = root / "reference_embeddings.npy"; np.save(refs, np.ones((8, 384), dtype=np.float32))
            fake_package = SimpleNamespace(SentenceTransformer=LocalOnlyStub)
            with patch.dict(sys.modules, {"sentence_transformers": fake_package}):
                model = ReferenceSBERT(model_dir=model_path, reference_embeddings_path=refs, require_local=True)
            self.assertEqual(model.model.path, str(model_path))
            self.assertTrue(model.model.kwargs["local_files_only"])

    def test_end_to_end_cache_evaluation_metadata_and_semantic_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, course_file, raw_courses, manual, output = (root / "reviews.csv", root / "courses.csv",
                root / "courses_raw.csv", root / "manual.csv", root / "output")
            pd.DataFrame({"review": ["positive one", "negative one", "neutral one", "positive two"],
                          "platform": ["site", "site", "", "site"], "rating": [5, 1, 3, "bad"],
                          "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]}).to_csv(source, index=False)
            pd.DataFrame({"title": ["Course X"], "rating": [4.7], "price": [12], "subscribers": [300]}).to_csv(course_file, index=False)
            pd.DataFrame({"title": ["Course Y"], "rating": [4.2], "price": ["Free"], "subscribers": [50]}).to_csv(raw_courses, index=False)
            pd.DataFrame({"review": ["positive label", "negative label", "neutral label"],
                          "label": ["positive", "negative", "neutral"]}).to_csv(manual, index=False)
            analyzer = FakeAnalyzer()
            with patch.object(run_analysis, "ROOT", root):
                first = run_analysis.run_analysis(str(source), str(output), str(course_file), str(manual), chunk_size=2, analyzer=analyzer, courses_raw_path=str(raw_courses))
                embedded_after_first = FakeAnalyzer.embedded
                with patch.object(run_analysis, "ReferenceSBERT", side_effect=AssertionError("cached run must not load SBERT")):
                    second = run_analysis.run_analysis(str(source), str(output), str(course_file), str(manual), chunk_size=2, courses_raw_path=str(raw_courses))
                self.assertEqual(FakeAnalyzer.embedded, embedded_after_first)
                self.assertEqual(first, second)
                original_predictions = pd.read_csv(first / "review_predictions.csv")
                with source.open("a", encoding="utf-8") as stream:
                    stream.write('"positive appended",site,5,2024-01-04\n')
                before_append = FakeAnalyzer.embedded
                appended = run_analysis.run_analysis(str(source), str(output), str(course_file), str(manual), chunk_size=2, analyzer=analyzer, courses_raw_path=str(raw_courses))
                self.assertEqual(FakeAnalyzer.embedded - before_append, 1)
                self.assertEqual(len(pd.read_csv(appended / "review_predictions.csv")), 5)
                before_rebuild = FakeAnalyzer.embedded
                rebuilt = run_analysis.run_analysis(str(source), str(output), str(course_file), str(manual), chunk_size=2,
                    analyzer=analyzer, courses_raw_path=str(raw_courses), rebuild=True)
                self.assertNotEqual(appended, rebuilt)
                self.assertEqual(FakeAnalyzer.embedded - before_rebuild, 8)
            self.assertTrue((root / "models/all-MiniLM-L6-v2/local_model.marker").is_file())
            self.assertTrue((root / "models/reference_embeddings.npy").is_file())
            final_predictions = pd.read_csv(first / "review_predictions.csv")
            self.assertEqual(len(final_predictions), 5)
            self.assertEqual(final_predictions.iloc[:4].review_text.tolist(), original_predictions.review_text.tolist())
            self.assertEqual(len(pd.read_csv(appended / "review_predictions.csv")), 5)
            metrics = json.loads((first / "evaluation_metrics.json").read_text())
            self.assertEqual(metrics["evaluation"]["accuracy"], 1.0)
            self.assertEqual(json.loads((first / "sentiment_summary.json").read_text())["review_count"], 5)
            self.assertTrue((first / "course_analysis.csv").is_file())
            self.assertEqual(pd.read_csv(first / "course_analysis_raw.csv").iloc[0].price, 0)
            self.assertTrue((first / "sentiment_trends.csv").is_file())
            matches = retrieve_reviews("positive", analyzer, first / "review_predictions.csv",
                                       Path(json.loads((first / "analysis_report.json").read_text())["embedding_cache_directory"]), top_k=2)
            self.assertEqual(len(matches), 2)
            for artifact in ("sentiment_distribution.png", "platform_sentiment.png", "rating_vs_sentiment.png",
                             "top_rated_courses.png", "top_subscribed_courses.png",
                             "top_rated_courses_raw.png", "course_price_distribution_raw.png"):
                self.assertTrue((first / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
