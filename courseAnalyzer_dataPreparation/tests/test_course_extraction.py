import json
import socket

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import database, extractors, main, security

PUBLIC_IP = [(None, None, None, None, ("93.184.216.34", 443))]


def embedded_review(review_id, comment, rating=5, name="AB", timestamp="2021-01-02T00:00:00.000Z", typename="DescriptionPage_Review"):
    return json.dumps({"__typename": typename, "id": review_id, "authorName": name,
                       "comment": comment, "rating": rating, "timestamp": timestamp})


def course_page(*records):
    apollo = ",".join(f'"k{i}":{record}' for i, record in enumerate(records))
    return ('<html><title>Python Basics</title><script type="application/ld+json">'
            '{"@type":"Course","name":"Python Basics","description":"Learn Python",'
            '"provider":{"name":"Test University"},'
            '"aggregateRating":{"ratingValue":4.8,"ratingCount":233779}}</script>'
            f'<script>window.__APOLLO_STATE__ = {{{apollo}}}</script></html>')


def test_coursera_parser_keeps_real_reviews_and_reports_skips():
    html = course_page(
        embedded_review("1", "  Great course, very clear.  ", rating=5),
        embedded_review("1", "Duplicate of the first review", rating=5),
        embedded_review("2", "   ", rating=4),
        embedded_review("3", "Useful but no rating given", rating=None),
    )
    result = extractors.parse_coursera_reviews(html, "https://www.coursera.org/learn/python")
    assert [r["review_id"] for r in result["reviews"]] == ["coursera:1", "coursera:3"]
    assert result["reviews"][0]["review_text"] == "Great course, very clear."
    assert result["reviews"][0]["rating"] == 5.0
    assert result["reviews"][1]["rating"] is None
    assert result["skipped_reviews"] == {"duplicate_on_page": 1, "empty_text": 1}
    assert result["reviews"]
    assert "robots.txt" in result["review_status"]


def test_page_without_reviews_is_reported_as_not_public():
    result = extractors.parse_coursera_reviews(course_page(), "https://www.coursera.org/learn/python")
    assert result["reviews"] == []
    assert "No public review text" in result["review_status"]


def test_unsupported_platform_never_reports_reviews(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: PUBLIC_IP)
    html = course_page(embedded_review("9", "Should not be used for udemy"))

    class Response:
        status_code = 200
        headers = {}
        text = html
        def raise_for_status(self): pass

    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url):
            return Response() if not url.endswith("robots.txt") else type("R", (), {"status_code": 404, "text": ""})()

    monkeypatch.setattr(extractors.httpx, "Client", Client)
    result = extractors.UdemyExtractor().extract("https://www.udemy.com/course/x")
    assert result["reviews"] == []
    assert result["reviews_retrieved"] == 0


class FakeModel:
    def embed(self, texts):
        result = np.zeros((len(texts), 384), dtype=np.float32)
        result[:, 0] = 1
        return result

    def classify_embeddings(self, vectors):
        count = len(vectors)
        return np.tile([0.8, 0.1, 0.1], (count, 1)), np.array(["positive"] * count), np.full(count, 0.7), np.zeros(count, dtype=bool)

    def save_model(self, path):
        pass

    def save_reference_embeddings(self, path):
        pass


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
        test_client.post("/auth/register", json={"email": "extract@example.test", "display_name": "e",
                         "password": "long-password-123", "password_confirmation": "long-password-123"})
        yield test_client


def use_extractor_result(monkeypatch, result):
    class StubExtractor:
        def extract(self, url):
            return json.loads(json.dumps(result))
    monkeypatch.setitem(main.EXTRACTORS, "coursera", StubExtractor)


def test_analyze_persists_extracted_reviews_and_deduplicates_resubmission(client, monkeypatch):
    reviews = [
        {"review_id": "coursera:1", "review_text": "Great course", "rating": 5.0, "review_date": "2021-01-02T00:00:00.000Z",
         "reviewer": "AB", "platform": "coursera", "source_url": "https://www.coursera.org/learn/python", "extracted_at": "2026-01-01T00:00:00+00:00"},
        {"review_id": "coursera:2", "review_text": "Too fast", "rating": None, "review_date": None,
         "reviewer": "CD", "platform": "coursera", "source_url": "https://www.coursera.org/learn/python", "extracted_at": "2026-01-01T00:00:00+00:00"},
    ]
    use_extractor_result(monkeypatch, {"title": "Python Basics", "platform": "coursera", "url": "https://www.coursera.org/learn/python",
                                       "extraction_status": "reviews_retrieved", "rating": 4.8, "review_count": 233779,
                                       "reviews": reviews, "reviews_retrieved": 2, "skipped_reviews": {},
                                       "review_status": "Retrieved 2 public reviews."})
    first = client.post("/courses/analyze", json={"url": "https://www.coursera.org/learn/python"})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["reviews_analyzed"] == 2
    assert body["review_status"] == "extracted_and_analyzed"
    assert body["extraction_message"] == "Retrieved 2 public reviews."
    assert body["summary"]["review_count"] == 2
    assert body["new_reviews_analyzed"] == 2
    second = client.post("/courses/analyze", json={"url": "https://www.coursera.org/learn/python"})
    assert second.status_code == 200, second.text
    assert second.json()["id"] == body["id"]
    assert second.json()["new_reviews_analyzed"] == 0
    assert len(client.get("/courses").json()) == 1
    stored = pd.read_csv(main.DATA / body["id"] / "review_predictions.csv")
    assert len(stored) == 2
    assert (stored.review_key == "coursera:1").sum() == 1
    assert {"source_url", "extracted_at", "review_date", "rating"}.issubset(stored.columns)
    assert stored.loc[stored.review_key == "coursera:2", "rating"].isna().all()


def test_analyze_reports_unavailable_reviews_instead_of_zero_success(client, monkeypatch):
    use_extractor_result(monkeypatch, {"title": "Course", "platform": "coursera", "url": "https://www.coursera.org/learn/x",
                                       "extraction_status": "metadata_only", "reviews": [], "reviews_retrieved": 0,
                                       "skipped_reviews": {}, "review_status": "No public review text was embedded."})
    response = client.post("/courses/analyze", json={"url": "https://www.coursera.org/learn/x"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["extraction_status"] == "metadata_only"
    assert body["reviews_analyzed"] == 0
    assert body["extraction_message"] == "No public review text was embedded."


@pytest.mark.network
def test_live_coursera_public_page_returns_real_reviews():
    """Integration check against a publicly accessible Coursera course page (requires network)."""
    try:
        result = extractors.CourseraExtractor().extract("https://www.coursera.org/learn/python")
    except Exception as exc:
        pytest.skip(f"Public page not reachable: {type(exc).__name__}")
    assert result["title"]
    assert result["platform"] == "coursera"
    assert result["rating"] is not None
    if result["extraction_status"] == "metadata_only":
        pytest.skip("Coursera page no longer embeds review text; extractor reported it honestly.")
    assert result["reviews_retrieved"] == len(result["reviews"]) > 0
    assert all(item["review_text"] and item["platform"] == "coursera" for item in result["reviews"])
