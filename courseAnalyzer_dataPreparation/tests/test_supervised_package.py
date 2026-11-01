import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import database, main
from src import portable_inference

PACKAGE = Path(__file__).resolve().parents[1] / "outputs" / "models"


class FakeEncoder:
    def embed(self, texts):
        return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))


class FakeClassifier:
    classes_ = np.array(["negative", "neutral", "positive"])

    def predict(self, features):
        return np.array(["positive"] * len(features))

    def predict_proba(self, features):
        return np.tile([0.1, 0.2, 0.7], (len(features), 1))


def test_predict_with_predictor_reuses_loaded_components_without_reading_disk():
    predictor = (FakeEncoder(), {"logistic_regression": FakeClassifier()}, {"default_classifier": "logistic_regression"})
    rows = portable_inference.predict_with_predictor(predictor, ["<b>Great</b> course", "bad"])
    assert [row["predicted_label"] for row in rows] == ["positive", "positive"]
    assert rows[0]["cleaned_text"] == "Great course"
    assert rows[0]["classifier"] == "logistic_regression"
    assert rows[0]["class_scores"] == {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
    with pytest.raises(ValueError):
        portable_inference.predict_with_predictor(predictor, ["x"], classifier="missing")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_ANALYZER_SECRET", "test-secret-with-enough-entropy-1234567890")
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.sqlite3")
    monkeypatch.setattr(main, "ROOT", tmp_path)
    with TestClient(main.app) as test_client:
        test_client.post("/auth/register", json={"email": "g@example.test", "display_name": "g",
                         "password": "long-password-123", "password_confirmation": "long-password-123"})
        yield test_client


def test_model_status_is_unavailable_and_never_trains_without_package(client):
    status = client.get("/model/status").json()
    assert status["available"] is False
    assert status["training_runs_automatically"] is False
    assert status["graphs"] == []


def test_graph_endpoint_serves_only_png_files_inside_graphs_directory(client, tmp_path):
    graphs = tmp_path / "outputs" / "training" / "graphs"
    graphs.mkdir(parents=True)
    (graphs / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    assert client.get("/model/graphs/chart.png").status_code == 200
    assert client.get("/model/graphs/..%2Fsecret.txt").status_code == 404
    assert client.get("/model/graphs/secret.txt").status_code == 404


@pytest.mark.skipif(not (PACKAGE / "model_metadata.json").is_file(), reason="packaged model not built yet")
def test_packaged_model_is_portable_relative_and_predicts():
    metadata = json.loads((PACKAGE / "model_metadata.json").read_text(encoding="utf-8"))
    serialized = json.dumps(metadata)
    assert ":\\" not in serialized and "/home/" not in serialized and "/Users/" not in serialized
    assert all(not Path(value).is_absolute() for value in metadata["paths"].values())
    rows = portable_inference.predict_texts(["The flavour is excellent and I would buy it again.",
                                             "This product was disappointing and poor quality."], PACKAGE)
    assert len(rows) == 2
    assert all(row["predicted_label"] in {"negative", "neutral", "positive"} for row in rows)
