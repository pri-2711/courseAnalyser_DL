import json
import socket

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import backfill, database, extractors, main, recommender

PUBLIC_IP = [(None, None, None, None, ("93.184.216.34", 443))]

ARTICLE_HTML = """<html><head><title>Fallback title</title>
<meta property="og:title" content="What Is Python?">
<meta property="og:description" content="A beginner overview of Python.">
<meta property="og:site_name" content="freeCodeCamp">
<meta name="author" content="Jane Doe">
<script type="application/ld+json">{"@context":"https://schema.org","@type":"NewsArticle",
"headline":"What Is Python?","description":"Python explained for beginners.",
"author":{"@type":"Person","name":"Jane Doe"},"publisher":{"@type":"Organization","name":"freeCodeCamp"}}</script>
</head><body><h2>Install Python</h2><h2>Write your first program</h2><h3>Variables</h3></body></html>"""

BARE_HTML = "<html><head><title>Plain page</title></head><body><p>Nothing structured here.</p></body></html>"


def test_article_json_ld_and_outline_are_extracted_without_fabrication():
    result = extractors.parse_page_metadata(ARTICLE_HTML, "https://www.freecodecamp.org/news/what-is-python/", "generic")
    assert result["title"] == "What Is Python?"
    assert result["description"] == "Python explained for beginners."
    assert result["provider"] == "freeCodeCamp"
    assert result["instructor"] == "Jane Doe"
    assert result["resource_type"] == "article"
    assert result["learning_outline"] == ["Install Python", "Write your first program", "Variables"]
    assert result["rating"] is None and result["review_count"] is None and result["duration"] is None
    assert {"rating", "review_count", "duration", "price"}.issubset(result["fields_unavailable"])
    assert result["metadata_extracted"] is True


def test_open_graph_fallback_when_no_structured_data():
    html = """<html><head><meta property="og:title" content="Tutorial X">
    <meta name="description" content="Short description."></head></html>"""
    result = extractors.parse_page_metadata(html, "https://example.org/x", "generic")
    assert result["title"] == "Tutorial X" and result["description"] == "Short description."
    assert result["learning_outline"] == []


def test_page_title_alone_counts_as_metadata_and_empty_page_fails():
    bare = extractors.GenericCourseExtractor()._result("https://example.org/bare", BARE_HTML)
    assert bare["extraction_status"] == "metadata_only" and bare["title"] == "Plain page"
    assert bare["description"] is None and "description" in bare["fields_unavailable"]


def test_status_distinguishes_metadata_only_reviews_and_failure():
    extractor = extractors.GenericCourseExtractor()
    metadata_only = extractor._result("https://www.freecodecamp.org/news/x/", ARTICLE_HTML)
    assert metadata_only["extraction_status"] == "metadata_only"
    assert metadata_only["reviews_retrieved"] == 0
    assert "No public review text" in metadata_only["review_status"]
    empty = extractor._result("https://example.org/empty", "<html></html>")
    assert empty["extraction_status"] == "extraction_failed"


def test_ranking_orders_by_relevance_and_explains_missing_signals():
    query = np.array([1.0, 0.0], dtype=np.float32)

    def resource(identifier, relevance_vector, **extra):
        base = {"id": identifier, "title": identifier, "provider": None, "platform": "generic", "url": "https://x",
                "description": None, "resource_type": "course", "rating": None, "review_count": None,
                "reviews_analyzed": 0, "sentiment_counts": None, "sentiment_value": None,
                "vector": np.asarray(relevance_vector, dtype=np.float32)}
        base.update(extra)
        return base

    ranked = recommender.rank(query, [
        resource("weak", [0.3, 0.9539392]),
        resource("strong", [0.9, 0.4358899]),
        resource("unrelated", [0.0, 1.0]),
        resource("rated", [0.9, 0.4358899], rating=5.0, review_count=100, reviews_analyzed=10, sentiment_value=1.0),
    ])
    assert [item["id"] for item in ranked] == ["rated", "strong", "weak"]
    assert all(item["relevance"] >= recommender.MIN_RELEVANCE for item in ranked)
    strong = next(item for item in ranked if item["id"] == "strong")
    assert "no rating published" in strong["explanation"] and "no analysed reviews" in strong["explanation"]
    rated = ranked[0]
    assert rated["score"] > strong["score"]
    assert "rating 5.00/5" in rated["explanation"] and "sentiment value +1.00" in rated["explanation"]


def test_resource_vector_is_cached_and_recomputed_only_when_text_changes(tmp_path):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32) / 2

    first = recommender.cached_resource_vector(tmp_path, "python course", encode)
    second = recommender.cached_resource_vector(tmp_path, "python course", encode)
    assert len(calls) == 1 and np.allclose(first, second)
    recommender.cached_resource_vector(tmp_path, "new text", encode)
    assert len(calls) == 2


def test_resource_text_skips_missing_fields():
    assert recommender.resource_text({}) == ""
    assert recommender.resource_text({"title": "Only title", "description": None}) == "Only title"


def test_backfill_fills_only_blank_cells_and_refuses_mismatch(tmp_path):
    class Stub:
        def predict(self, vectors):
            return np.array(["positive"] * len(vectors), dtype=object)

    embeddings = tmp_path / "embeddings.npy"
    np.save(embeddings, np.zeros((3, 384), dtype=np.float16))
    predictions = tmp_path / "review_predictions.csv"
    frame = pd.DataFrame({"review_text": ["a", "b", "c"], "sentiment_label": ["negative", "neutral", "positive"],
                          "embedding_row": [0, 1, 2], "mlp_label": ["neutral", None, None]})
    frame.to_csv(predictions, index=False)
    filled, report = backfill.backfill_supervised_labels(frame, predictions, embeddings, {"mlp": Stub()})
    assert report["status"] == "filled" and report["filled"] == {"mlp_label": 2}
    assert filled["mlp_label"].tolist() == ["neutral", "positive", "positive"]
    assert filled["sentiment_label"].tolist() == ["negative", "neutral", "positive"]
    bad = frame.assign(embedding_row=[0, 1, 9])
    unchanged, report = backfill.backfill_supervised_labels(bad, predictions, embeddings, {"mlp": Stub()})
    assert report["status"] == "embedding_mismatch" and unchanged is bad


class FakeModel:
    def embed(self, texts):
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0] = 1.0 if "python" in text.lower() else 0.0
            vectors[i, 1] = 1.0 if "cooking" in text.lower() else 0.0
            vectors[i, 2] = 0.1
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_ANALYZER_SECRET", "test-secret-with-enough-entropy-1234567890")
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.sqlite3")
    monkeypatch.setattr(main, "DATA", tmp_path / "courses")
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "MODEL", FakeModel())
    monkeypatch.setattr(main, "_model", lambda: main.MODEL)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: PUBLIC_IP)
    with TestClient(main.app) as test_client:
        test_client.post("/auth/register", json={"email": "rec@example.test", "display_name": "r",
                         "password": "long-password-123", "password_confirmation": "long-password-123"})
        yield test_client


def add_resource(client, title, description, url, provider="Test Provider"):
    with database.connect() as db:
        user_id = db.execute("SELECT id FROM users WHERE email='rec@example.test'").fetchone()["id"]
    metadata = {"title": title, "description": description, "platform": "generic", "url": url,
                "provider": provider, "rating": None, "review_count": None, "metadata_extracted": True}
    return main._write_analysis(user_id, url, metadata, {}, "", "")


def test_recommend_validates_query_and_handles_empty_catalogue(client):
    assert client.post("/recommendations", json={"query": "ab"}).status_code == 422
    assert client.post("/recommendations", json={"query": "   "}).status_code == 422
    empty = client.post("/recommendations", json={"query": "python basics"}).json()
    assert empty["status"] == "empty_catalogue" and empty["results"] == []


def test_recommend_ranks_saved_resources_and_links_originals(client):
    python_id = add_resource(client, "Python for Beginners", "Learn python programming basics", "https://example.org/python")
    cooking_id = add_resource(client, "Cooking Basics", "Learn cooking recipes", "https://example.org/cooking")
    add_resource(client, None, None, "https://example.org/empty", provider=None)
    body = client.post("/recommendations", json={"query": "python programming"}).json()
    assert body["status"] == "ok"
    assert body["results"][0]["id"] == python_id
    assert body["results"][0]["url"] == "https://example.org/python"
    assert all(item["id"] != cooking_id for item in body["results"])
    assert [item["url"] for item in body["skipped"]] == ["https://example.org/empty"]
    assert "cannot be matched" in body["skipped"][0]["reason"]
    assert "semantic similarity" in body["results"][0]["explanation"]
    assert "vector" not in body["results"][0]


def test_model_breakdown_reports_each_classifier_separately():
    frame = pd.DataFrame({"sentiment_label": ["positive", "negative", "neutral", "positive"],
                          "mlp_label": ["positive", None, "neutral", None],
                          "linear_svm_label": [None, None, None, None]})
    breakdown = main._model_breakdown(frame)
    assert breakdown["sbert_reference"]["counts"] == {"positive": 2, "negative": 1, "neutral": 1}
    assert breakdown["mlp"]["total"] == 2 and breakdown["mlp"]["rows_without_prediction"] == 2
    assert breakdown["mlp"]["counts"]["negative"] == 0
    assert breakdown["linear_svm"]["available"] is False

def test_iso_duration_and_person_only_instructor():
    assert extractors._duration("PT24H") == "24 hours"
    assert extractors._duration("PT1H30M") == "1 hour 30 minutes"
    assert extractors._duration("4 weeks") == "4 weeks"
    assert extractors._duration(None) is None
    assert extractors._person({"@type": "Person", "name": "Ada"}) == "Ada"
    assert extractors._person({"@type": "Organization", "name": "GeeksforGeeks"}) is None


def test_access_check_page_is_not_presented_as_course_metadata():
    challenge = "<html><head><title>Client Challenge</title></head><body><h2>Verifying</h2></body></html>"
    result = extractors.GenericCourseExtractor()._result("https://www.khanacademy.org/x", challenge)
    assert result["extraction_status"] == "access_restricted"
    assert result["title"] is None and result["reviews"] == []
    article = "<html><head><title>How captchas work</title></head><body><p>Captcha basics.</p></body></html>"
    assert extractors.GenericCourseExtractor()._result("https://example.org/a", article)["extraction_status"] == "metadata_only"


def test_mlp_is_the_application_default_classifier():
    from src.portable_inference import DISPLAY_DEFAULT_CLASSIFIER, predict_with_predictor

    class Encoder:
        def embed(self, texts):
            return np.ones((len(texts), 2), dtype=np.float32)

    class Model:
        classes_ = np.asarray(["negative", "neutral", "positive"])

        def __init__(self, label):
            self.label = label

        def predict(self, features):
            return np.array([self.label] * len(features))

    predictor = (Encoder(), {"mlp": Model("positive"), "linear_svm": Model("negative")}, {"default_classifier": "linear_svm"})
    assert DISPLAY_DEFAULT_CLASSIFIER == "mlp"
    rows = predict_with_predictor(predictor, ["a review"])
    assert rows[0]["classifier"] == "mlp" and rows[0]["predicted_label"] == "positive"
    assert predict_with_predictor(predictor, ["a review"], "linear_svm")[0]["predicted_label"] == "negative"
