import io
import json
import socket
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import database, extractors, main, security


class FakeModel:
    calls = 0

    def embed(self, texts):
        type(self).calls += len(texts)
        result = np.zeros((len(texts), 384), dtype=np.float32)
        result[:, 0] = 1
        return result

    def classify_embeddings(self, vectors):
        scores = np.tile([0.8, 0.1, 0.1], (len(vectors), 1))
        count = len(vectors)
        return scores, np.array(["positive"] * count), np.full(count, 0.7), np.zeros(count, dtype=bool)

    def classify(self, texts):
        return self.classify_embeddings(self.embed(texts))

    def save_model(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.marker").write_text("local", encoding="utf-8")

    def save_reference_embeddings(self, path):
        np.save(path, np.ones((8, 384), dtype=np.float32))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_ANALYZER_SECRET", "test-secret-with-enough-entropy-1234567890")
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.sqlite3")
    monkeypatch.setattr(main, "DATA", tmp_path / "courses")
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "MODEL", FakeModel())
    FakeModel.calls = 0
    with TestClient(main.app) as test_client:
        yield test_client


def register(client, email):
    response = client.post("/auth/register", json={"email": email, "display_name": email.split("@")[0],
        "password": "long-password-123", "password_confirmation": "long-password-123"})
    assert response.status_code == 200, response.text
    return response.json()


def test_bcrypt_hash_and_signed_session(monkeypatch):
    monkeypatch.setenv("COURSE_ANALYZER_SECRET", "a-long-test-secret-that-is-unpredictable")
    hashed = security.hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert security.verify_password("correct horse battery staple", hashed)
    assert not security.verify_password("incorrect", hashed)
    session_id, _, cookie = security.new_session()
    assert security.verify_session(cookie) == session_id
    assert security.verify_session(cookie + "tampered") is None
    expired = security.sign_session("expired-id", int(time.time()) - 1)
    assert security.verify_session(expired) is None


def test_register_protected_endpoint_ownership_and_logout(client):
    first = register(client, "one@example.test")
    protected = client.get("/courses")
    assert protected.status_code == 200
    row_id = main._write_analysis(first["id"], "https://udemy.com/course/a", {"title": "A", "platform": "udemy"}, {}, "", "")
    second = client.post("/auth/logout")
    assert second.status_code == 200
    assert client.get("/courses").status_code == 401
    register(client, "two@example.test")
    assert client.get(f"/courses/{row_id}").status_code == 404
    assert client.post("/courses/compare", json={"course_ids": [row_id, "missing"]}).status_code == 404


def test_registration_duplicate_password_confirmation_and_login(client):
    register(client, "learner@example.test")
    duplicate = client.post("/auth/register", json={"email": "learner@example.test", "display_name": "Duplicate",
        "password": "long-password-123", "password_confirmation": "long-password-123"})
    assert duplicate.status_code == 409
    mismatch = client.post("/auth/register", json={"email": "other@example.test", "display_name": "Learner",
        "password": "long-password-123", "password_confirmation": "different-password"})
    assert mismatch.status_code == 422
    client.post("/auth/logout")
    response = client.post("/auth/login", json={"email": "learner@example.test", "password": "long-password-123"})
    assert response.status_code == 200
    assert client.get("/auth/me").json()["email"] == "learner@example.test"


def test_login_rate_limit(client):
    for _ in range(8):
        assert client.post("/auth/login", json={"email": "missing@example.test", "password": "wrong"}).status_code == 401
    assert client.post("/auth/login", json={"email": "missing@example.test", "password": "wrong"}).status_code == 429


def test_url_validation_and_platform_detection(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))])
    assert extractors.detect_platform("https://www.udemy.com/course/example/") == "udemy"
    assert extractors.detect_platform("https://www.coursera.org/learn/example") == "coursera"
    assert extractors.detect_platform("https://www.edx.org/learn/example") == "edx"
    assert extractors.detect_platform("https://example.org/course") == "generic"
    for url in ("http://example.org", "https://localhost/course", "https://127.0.0.1/course"):
        with pytest.raises(extractors.UnsafeURL):
            extractors.validate_url(url)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(None, None, None, None, ("10.0.0.1", 443))])
    with pytest.raises(extractors.UnsafeURL):
        extractors.validate_url("https://rebind.example/course")


def test_platform_extractors_metadata_only_with_mocked_response(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))])

    class Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = '<html><title>Example</title><script type="application/ld+json">{"@type":"Course","name":"Test Course","description":"Learning","aggregateRating":{"ratingValue":"4.5","reviewCount":"12"}}</script></html>'
        def raise_for_status(self): pass

    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url): return Response()

    monkeypatch.setattr(extractors.httpx, "Client", Client)
    for extractor, url in ((extractors.UdemyExtractor(), "https://udemy.com/course/x"),
                           (extractors.CourseraExtractor(), "https://coursera.org/learn/x"),
                           (extractors.EdXExtractor(), "https://edx.org/course/x"),
                           (extractors.GenericCourseExtractor(), "https://example.org/course/x")):
        result = extractor.extract(url)
        assert result["title"] == "Test Course"
        assert result["reviews_retrieved"] == 0
        assert result["extraction_status"] == "metadata_only"
        assert result["review_count"] == 12
        assert result["fields_unavailable"] == ["reviews"] or "reviews" not in result["fields_unavailable"]


def test_robots_disallow_prevents_page_request(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))])
    requested = []

    class Response:
        status_code = 200
        headers = {}
        text = "User-agent: CourseAnalyzerLocal\nDisallow: /course/"
        def raise_for_status(self): pass

    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url): requested.append(url); return Response()

    monkeypatch.setattr(extractors.httpx, "Client", Client)
    result = extractors.GenericCourseExtractor().extract("https://example.org/course/x")
    assert result["extraction_status"] == "robots_disallowed"
    assert requested == ["https://example.org/robots.txt"]


def test_analysis_upload_cache_incremental_rebuild_and_evidence(client, monkeypatch):
    user = register(client, "analysis@example.test")
    monkeypatch.setattr(main, "_model", lambda: main.MODEL)
    course_id = main._write_analysis(user["id"], "https://udemy.com/course/x", {"title": "Course", "platform": "udemy"}, {}, "", "")
    first = client.post(f"/courses/{course_id}/reviews", files={"file": ("reviews.csv", b"review_id,review,rating\n1,Wonderful course,5\n2,Very useful,4\n", "text/csv")})
    assert first.status_code == 200, first.text
    assert first.json()["summary"]["review_count"] == 2
    assert FakeModel.calls == 2
    same = client.post(f"/courses/{course_id}/reviews", files={"file": ("reviews.csv", b"review_id,review,rating\n1,Wonderful course,5\n2,Very useful,4\n", "text/csv")})
    assert same.json()["cache_reused"] is True
    assert FakeModel.calls == 2
    appended = client.post(f"/courses/{course_id}/reviews", files={"file": ("reviews.csv", b"review_id,review,rating\n1,Wonderful course,5\n2,Very useful,4\n3,Great new material,5\n", "text/csv")})
    assert appended.status_code == 200, appended.text
    assert appended.json()["new_reviews_analyzed"] == 1
    assert FakeModel.calls == 3
    record = client.get(f"/courses/{course_id}").json()
    assert "reviews_path" not in record and "user_id" not in record
    evidence = client.post(f"/courses/{course_id}/ask", json={"question": "What did students like?"})
    assert evidence.status_code == 200 and evidence.json()["evidence"]
    evaluation = client.post(f"/courses/{course_id}/evaluation", files={"file": ("labels.csv", b"review,label\nExcellent course,positive\nNot for me,negative\n", "text/csv")})
    assert evaluation.status_code == 200, evaluation.text
    assert evaluation.json()["evaluation"]["sample_count"] == 2
    assert evaluation.json()["evaluation"]["accuracy"] == 0.5
    assert len(evaluation.json()["evaluation"]["misclassified_examples"]) == 1
    assert client.get(f"/courses/{course_id}/analytics").json()["evaluation"]["sample_count"] == 2
    rebuilt = client.post(f"/courses/{course_id}/rebuild", files={"file": ("reviews.csv", b"review_id,review,rating\n1,Wonderful course,5\n2,Very useful,4\n", "text/csv")})
    assert rebuilt.status_code == 200 and rebuilt.json()["new_reviews_analyzed"] == 2
    assert FakeModel.calls == 8
    assert (Path(main.ROOT) / "models" / "all-MiniLM-L6-v2" / "model.marker").exists()


def test_invalid_upload_and_delete_protect_course_data(client):
    user = register(client, "files@example.test")
    course_id = main._write_analysis(user["id"], "https://udemy.com/course/x", {"title": "Course", "platform": "udemy"}, {}, "", "")
    response = client.post(f"/courses/{course_id}/reviews", files={"file": ("bad.exe", b"no", "application/octet-stream")})
    assert response.status_code == 415
    assert client.delete(f"/courses/{course_id}").json()["deleted"] is True
    assert client.get(f"/courses/{course_id}").status_code == 404
