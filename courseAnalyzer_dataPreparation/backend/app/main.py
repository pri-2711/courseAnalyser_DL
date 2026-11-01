"""FastAPI application for local, account-scoped course review analysis."""
from contextlib import asynccontextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
from collections import Counter

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import normalize_reviews
from src.sbert_model import ReferenceSBERT
from . import database, recommender
from .backfill import backfill_supervised_labels
from .extractors import EXTRACTORS, METADATA_FIELDS, UnsafeURL, detect_platform, validate_url
from src.evaluation import EvaluationAccumulator, evaluation_columns, normalize_label
from src.preprocessing import clean_text
from src.portable_inference import DISPLAY_DEFAULT_CLASSIFIER, load_portable_predictor, predict_with_predictor
from .security import hash_password, new_session, verify_password, verify_session


@asynccontextmanager
async def lifespan(_app):
    database.initialize()
    try:
        # Loads the packaged classifier and encoder only; training is never started by the server.
        _supervised()
    except FileNotFoundError:
        pass  # Package not built yet; /model/status reports it as unavailable.
    yield


app = FastAPI(title="Course Analyzer Local API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("COURSE_ANALYZER_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
                   allow_credentials=True, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"])
ROOT = PROJECT_ROOT
DATA = Path(os.environ.get("COURSE_ANALYZER_DATA", ROOT / "data" / "courses"))
COOKIE_SECURE = os.environ.get("COURSE_ANALYZER_COOKIE_SECURE", "0") == "1"
MODEL = None
SUPERVISED = None
LOGIN_ATTEMPTS = {}
MAX_UPLOAD = 20 * 1024 * 1024


class RegisterBody(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=12, max_length=128)
    password_confirmation: str = Field(min_length=12, max_length=128)

    @field_validator("password_confirmation")
    @classmethod
    def passwords_match(cls, value, info):
        if "password" in info.data and value != info.data["password"]:
            raise ValueError("Passwords do not match.")
        return value

    @field_validator("email")
    @classmethod
    def valid_email(cls, value):
        value = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Enter a valid email address.")
        return value

    @field_validator("display_name")
    @classmethod
    def valid_display_name(cls, value):
        if not value.strip():
            raise ValueError("Display name cannot be empty.")
        return value.strip()

    @field_validator("password")
    @classmethod
    def bcrypt_password_length(cls, value):
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes for bcrypt.")
        return value


class LoginBody(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(max_length=72)


class AnalyzeBody(BaseModel):
    url: str = Field(min_length=10, max_length=2048)


class CompareBody(BaseModel):
    course_ids: list[str] = Field(min_length=2, max_length=10)


class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class PredictBody(BaseModel):
    reviews: list[str] = Field(min_length=1, max_length=100)
    classifier: str | None = None


def _owner(request: Request):
    token = request.cookies.get("course_session", "")
    session_id = verify_session(token)
    if not session_id:
        raise HTTPException(401, "Please sign in.")
    with database.connect() as db:
        row = db.execute("SELECT u.id,u.email,u.display_name,s.expires_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.id=?", (session_id,)).fetchone()
        if not row or row["expires_at"] <= int(time.time()):
            raise HTTPException(401, "Session expired. Please sign in again.")
        return {"id": row["id"], "email": row["email"], "display_name": row["display_name"]}


def _owned_analysis(analysis_id, user_id):
    with database.connect() as db:
        row = db.execute("SELECT * FROM analyses WHERE id=? AND user_id=?", (analysis_id, user_id)).fetchone()
    if not row:
        raise HTTPException(404, "Analysis not found.")
    return dict(row)


def _json_record(row):
    result = dict(row)
    return {"id": result["id"], "url": result["url"], "platform": result["platform"],
            "title": result["title"], "metadata": json.loads(result["metadata_json"]),
            "summary": json.loads(result["summary_json"]), "created_at": result["created_at"]}


def _write_analysis(user_id, url, metadata, summary, fingerprint, reviews_path, embeddings_path=None, analysis_id=None):
    analysis_id = analysis_id or str(uuid.uuid4())
    with database.connect() as db:
        db.execute("INSERT INTO analyses(id,user_id,url,platform,title,metadata_json,summary_json,fingerprint,reviews_path,embeddings_path) VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (analysis_id, user_id, url, metadata.get("platform", "unknown"), metadata.get("title") or "Untitled course",
                    json.dumps(metadata), json.dumps(summary), fingerprint, str(reviews_path), str(embeddings_path) if embeddings_path else None))
    return analysis_id


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _model():
    global MODEL
    if MODEL is None:
        model_path = ROOT / "models" / "all-MiniLM-L6-v2"
        refs = ROOT / "models" / "reference_embeddings.npy"
        MODEL = ReferenceSBERT(batch_size=64, model_dir=model_path, reference_embeddings_path=refs,
                               require_local=model_path.is_dir())
    return MODEL


def _supervised():
    global SUPERVISED
    if SUPERVISED is None:
        SUPERVISED = load_portable_predictor(ROOT / "outputs" / "models")
    return SUPERVISED


def _summary(frame):
    counts = frame.sentiment_label.value_counts().to_dict()
    total = int(len(frame))
    labels = ("positive", "negative", "neutral")
    numeric_rating = pd.to_numeric(frame.rating, errors="coerce")
    result = {"review_count": total, "sentiment_counts": {k: int(counts.get(k, 0)) for k in labels},
              "sentiment_distribution": {k: counts.get(k, 0) / total if total else 0 for k in labels},
              "sentiment_value": ((counts.get("positive", 0) - counts.get("negative", 0)) / total) if total else None,
              "average_review_rating": float(numeric_rating.mean()) if numeric_rating.notna().any() else None,
              "uncertain_review_count": int(frame.is_uncertain.sum())}
    rating = frame.assign(rating_bin=numeric_rating).dropna(subset=["rating_bin"]).groupby("rating_bin").sentiment_label.value_counts().unstack(fill_value=0)
    result["rating_sentiment"] = {str(key): {str(label): int(value) for label, value in row.items()}
                                   for key, row in rating.to_dict(orient="index").items()}
    stop = {"the", "and", "for", "with", "this", "that", "was", "were", "are", "is", "it", "to", "of", "in", "on", "a", "an", "my", "i", "you", "your", "but", "not", "have", "had", "has", "very", "be", "as", "at", "by", "from", "or", "after", "about"}
    phrases = {}
    for label in ("positive", "negative", "neutral"):
        counts_by_review = Counter()
        for text in frame.loc[frame.sentiment_label == label, "review_text"].fillna("").astype(str):
            words = [word for word in re.findall(r"[a-z]{3,}", text.lower()) if word not in stop]
            seen = sorted({" ".join(words[index:index + size]) for size in (2, 3)
                           for index in range(max(0, len(words) - size + 1))})
            counts_by_review.update(seen)
        phrases[label] = [{"phrase": phrase, "review_mentions": int(count)}
                          for phrase, count in counts_by_review.most_common(8)]
    result["frequent_phrases"] = phrases
    if "review_date" in frame:
        dated = frame.assign(_date=pd.to_datetime(frame.review_date, errors="coerce", utc=True).dt.strftime("%Y-%m-%d")).dropna(subset=["_date"])
        trend = dated.groupby(["_date", "sentiment_label"]).size().unstack(fill_value=0)
        result["review_trends"] = [{"date": str(day), "review_count": int(values.sum()),
            "sentiment_value": float((values.get("positive", 0) - values.get("negative", 0)) / values.sum())}
            for day, values in trend.iterrows() if values.sum()]
    else:
        result["review_trends"] = []
    result["model_breakdown"] = _model_breakdown(frame)
    result["default_model"] = DISPLAY_DEFAULT_CLASSIFIER
    return result


MODEL_SOURCES = (("sbert_reference", "sentiment_label", "SBERT reference-similarity classifier"),
                 ("mlp", "mlp_label", "Supervised PyTorch MLP (Deep Learning)"),
                 ("linear_svm", "linear_svm_label", "Linear SVM (Traditional ML)"))


def _model_breakdown(frame):
    """Per-classifier sentiment counts. Each classifier is reported separately; rows without that
    classifier's prediction (historical records) are excluded from its counts and reported."""
    breakdown = {}
    for key, column, label in MODEL_SOURCES:
        if column not in frame:
            breakdown[key] = {"model": label, "available": False, "total": 0, "rows_without_prediction": int(len(frame)),
                              "counts": {k: 0 for k in ("positive", "negative", "neutral")}, "percent": {}}
            continue
        labels = frame[column].where(frame[column].notna(), None)
        predicted = labels.dropna().astype(str)
        counts = {k: int((predicted == k).sum()) for k in ("positive", "negative", "neutral")}
        total = int(len(predicted))
        breakdown[key] = {"model": label, "available": total > 0, "total": total,
                          "rows_without_prediction": int(len(frame) - total), "counts": counts,
                          "percent": {k: (counts[k] / total if total else None) for k in counts}}
    return breakdown


class RecommendBody(BaseModel):
    query: str = Field(min_length=3, max_length=200)

    @field_validator("query")
    @classmethod
    def meaningful_query(cls, value):
        value = value.strip()
        if not re.search(r"[A-Za-z0-9]{2,}", value):
            raise ValueError("Enter a topic or skill with at least two letters or numbers.")
        return value


@app.post("/recommendations")
def recommend(body: RecommendBody, user=Depends(_owner)):
    with database.connect() as db:
        rows = db.execute("SELECT * FROM analyses WHERE user_id=? ORDER BY created_at DESC", (user["id"],)).fetchall()
    if not rows:
        return {"status": "empty_catalogue", "query": body.query, "results": [], "skipped": [],
                "message": "Your catalogue is empty. Add a course or resource URL on the dashboard, then search again.",
                "weights": recommender.RANKING_WEIGHTS}
    model = _model()
    candidates, skipped = [], []
    for row in rows:
        metadata, summary = recommender.metadata_from_row(row)
        text = recommender.resource_text(metadata)
        title = metadata.get("title") or row["url"]
        if not text:
            skipped.append({"id": row["id"], "title": title, "url": row["url"],
                            "reason": "No title or description was extracted, so this resource cannot be matched."})
            continue
        directory = DATA / row["id"]
        directory.mkdir(parents=True, exist_ok=True)
        vector = recommender.cached_resource_vector(directory, text, model.embed)
        candidates.append({"vector": vector, "id": row["id"], "title": title, "provider": metadata.get("provider"),
                           "platform": row["platform"], "url": row["url"], "description": metadata.get("description"),
                           "resource_type": metadata.get("resource_type"), "rating": metadata.get("rating"),
                           "review_count": metadata.get("review_count"),
                           "reviews_analyzed": int(metadata.get("reviews_analyzed") or 0),
                           "sentiment_counts": summary.get("sentiment_counts"),
                           "sentiment_value": summary.get("sentiment_value")})
    if not candidates:
        return {"status": "no_matchable_resources", "query": body.query, "results": [], "skipped": skipped,
                "message": "None of your saved resources has a title or description to match against.",
                "weights": recommender.RANKING_WEIGHTS}
    query_vector = np.asarray(model.embed([body.query])[0], dtype=np.float32)
    ranked = recommender.rank(query_vector, candidates)
    return {"status": "ok" if ranked else "no_strong_matches", "query": body.query, "results": ranked[:10],
            "skipped": skipped, "catalogue_size": len(rows), "weights": recommender.RANKING_WEIGHTS,
            "message": "" if ranked else f"No saved resource reached the minimum relevance of {recommender.MIN_RELEVANCE}. Add more resources by URL to widen the catalogue."}


@app.get("/health")
def health():
    return {"status": "ok", "model": "all-MiniLM-L6-v2", "review_extraction": "upload required unless a public source is configured"}


@app.get("/model/status")
def model_status(user=Depends(_owner)):
    package = ROOT / "outputs" / "models"
    metadata = package / "model_metadata.json"
    report = ROOT / "outputs" / "training" / "reports" / "training_report.json"
    result = {"default_classifier": DISPLAY_DEFAULT_CLASSIFIER, "available": metadata.is_file() and (package / "sentiment_model.joblib").is_file(),
              "training_runs_automatically": False}
    if metadata.is_file():
        result["metadata"] = json.loads(metadata.read_text(encoding="utf-8"))
    if report.is_file():
        document = json.loads(report.read_text(encoding="utf-8"))
        result["selected_model"] = document.get("selected_model")
        result["model_comparison"] = document.get("model_comparison")
    graphs_dir = ROOT / "outputs" / "training" / "graphs"
    result["graphs"] = sorted(item.name for item in graphs_dir.glob("*.png")) if graphs_dir.is_dir() else []
    return result


@app.get("/model/graphs/{name}")
def model_graph(name: str, user=Depends(_owner)):
    directory = (ROOT / "outputs" / "training" / "graphs").resolve()
    target = (directory / name).resolve()
    if target.parent != directory or target.suffix.lower() != ".png" or not target.is_file():
        raise HTTPException(404, "Graph not found")
    return FileResponse(target, media_type="image/png")


@app.post("/model/predict")
def model_predict(body: PredictBody, user=Depends(_owner)):
    try:
        return {"predictions": predict_with_predictor(_supervised(), body.reviews, body.classifier)}
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/auth/register")
def register(body: RegisterBody, response: Response):
    try:
        with database.connect() as db:
            cursor = db.execute("INSERT INTO users(email,password_hash,display_name) VALUES(?,?,?)",
                                (body.email.lower(), hash_password(body.password), body.display_name.strip()))
            user_id = cursor.lastrowid
    except Exception as exc:
        if "UNIQUE" in str(exc):
            raise HTTPException(409, "An account with this email already exists.")
        raise
    session_id, expires, cookie = new_session()
    with database.connect() as db:
        db.execute("INSERT INTO sessions(id,user_id,expires_at) VALUES(?,?,?)", (session_id, user_id, expires))
    response.set_cookie("course_session", cookie, httponly=True, secure=COOKIE_SECURE, samesite="lax", max_age=60 * 60 * 24 * 7, path="/")
    return {"id": user_id, "email": body.email.lower(), "display_name": body.display_name.strip()}


@app.post("/auth/login")
def login(body: LoginBody, request: Request, response: Response):
    key = (request.client.host if request.client else "local", body.email.lower())
    attempts, started = LOGIN_ATTEMPTS.get(key, (0, time.time()))
    if time.time() - started > 60:
        attempts, started = 0, time.time()
    if attempts >= 8:
        raise HTTPException(429, "Too many attempts. Try again in a minute.")
    with database.connect() as db:
        user = db.execute("SELECT * FROM users WHERE email=?", (body.email.lower(),)).fetchone()
    if not user or not verify_password(body.password, user["password_hash"]):
        LOGIN_ATTEMPTS[key] = (attempts + 1, started)
        raise HTTPException(401, "Email or password is incorrect.")
    LOGIN_ATTEMPTS.pop(key, None)
    session_id, expires, cookie = new_session()
    with database.connect() as db:
        db.execute("INSERT INTO sessions(id,user_id,expires_at) VALUES(?,?,?)", (session_id, user["id"], expires))
    response.set_cookie("course_session", cookie, httponly=True, secure=COOKIE_SECURE, samesite="lax", max_age=60 * 60 * 24 * 7, path="/")
    return {"id": user["id"], "email": user["email"], "display_name": user["display_name"]}


@app.post("/auth/logout")
def logout(request: Request, response: Response):
    session_id = verify_session(request.cookies.get("course_session", ""))
    if session_id:
        with database.connect() as db:
            db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    response.delete_cookie("course_session", path="/")
    return {"ok": True}


@app.get("/auth/me")
def me(user=Depends(_owner)):
    return user


@app.post("/courses/analyze")
def analyze(body: AnalyzeBody, user=Depends(_owner)):
    try:
        url = validate_url(body.url)
        platform = detect_platform(url)
        metadata = EXTRACTORS[platform]().extract(url)
    except (UnsafeURL, ValueError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        metadata = {"title": None, "platform": platform, "url": url, "extraction_status": "extraction_failed",
                    "metadata_extracted": False, "fields_unavailable": list(METADATA_FIELDS),
                    "reviews_retrieved": 0, "skipped_reviews": {},
                    "review_status": f"The course page could not be fetched ({type(exc).__name__}). Upload review data instead."}
    extracted = metadata.pop("reviews", [])
    metadata.setdefault("platform", platform)
    metadata["extraction_message"] = metadata.get("review_status")
    with database.connect() as db:
        existing = db.execute("SELECT id FROM analyses WHERE user_id=? AND url=? ORDER BY created_at LIMIT 1", (user["id"], url)).fetchone()
    if existing:
        # Resubmitting a URL refreshes the same course record instead of creating a duplicate.
        analysis_id = existing["id"]
        with database.connect() as db:
            db.execute("UPDATE analyses SET title=?,platform=?,metadata_json=? WHERE id=? AND user_id=?",
                       (metadata.get("title") or "Untitled course", metadata["platform"], json.dumps(metadata), analysis_id, user["id"]))
    else:
        analysis_id = _write_analysis(user["id"], url, metadata, {"review_count": 0, "sentiment_counts": {},
            "sentiment_distribution": {}, "sentiment_value": None, "status": "reviews_required"}, "", "")
    _atomic_json(DATA / analysis_id / "course_metadata.json", metadata)
    if not extracted:
        return {"id": analysis_id, **metadata, "reviews_analyzed": 0}
    # Extracted reviews use the same embedding, classification, de-duplication and persistence as uploads.
    frame = pd.DataFrame(extracted)
    digest = hashlib.sha256(json.dumps(extracted, sort_keys=True).encode("utf-8")).hexdigest()
    path = DATA / analysis_id / f"extracted_{digest}.json"
    _atomic_json(path, extracted)
    record = _owned_analysis(analysis_id, user["id"])
    try:
        result = _ingest_reviews(analysis_id, record, user, frame, normalize_reviews(frame), digest, path, False,
                                 status_label="extracted_and_analyzed")
        return {"id": analysis_id, **result["metadata"], "summary": result["summary"],
                "new_reviews_analyzed": result["new_reviews_analyzed"], "cache_reused": result["cache_reused"]}
    except HTTPException:
        raise
    except Exception as exc:
        metadata.update({"review_status": f"Reviews were found but analysis failed ({type(exc).__name__}: {exc})."})
        with database.connect() as db:
            db.execute("UPDATE analyses SET metadata_json=? WHERE id=? AND user_id=?", (json.dumps(metadata), analysis_id, user["id"]))
        raise HTTPException(500, metadata["review_status"])


@app.post("/courses/{analysis_id}/reviews")
async def upload_reviews(analysis_id: str, file: UploadFile = File(...), rebuild: bool = False, user=Depends(_owner)):
    record = _owned_analysis(analysis_id, user["id"])
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".json", ".jsonl", ".ndjson"}:
        raise HTTPException(415, "Upload a CSV, JSON or JSON Lines file.")
    content = await file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, "Review file exceeds the 20 MB local upload limit.")
    fingerprint = hashlib.sha256(content).hexdigest()
    if not rebuild and fingerprint == record["fingerprint"] and record["summary_json"]:
        return {"id": analysis_id, **_json_record(record), "cache_reused": True}
    directory = DATA / analysis_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("upload_" + fingerprint + suffix)
    path.write_bytes(content)
    try:
        if suffix == ".csv":
            raw = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
        elif suffix in {".jsonl", ".ndjson"} or content.lstrip().startswith(b"{") and b"\n{" in content:
            raw = pd.read_json(io.BytesIO(content), lines=True)
        else:
            raw = pd.read_json(io.BytesIO(content))
        reviews = normalize_reviews(raw)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Review file could not be read: {exc}")
    return _ingest_reviews(analysis_id, record, user, raw, reviews, fingerprint, path, rebuild)


def _ingest_reviews(analysis_id, record, user, raw, reviews, fingerprint, path, rebuild, status_label="uploaded_and_analyzed"):
    """Embed, classify and persist reviews for one course. Shared by uploads and URL extraction."""
    directory = DATA / analysis_id
    if reviews.empty:
        raise HTTPException(400, "No non-empty review text rows were found.")
    try:
        identity_col = next((c for c in ("review_id", "_id", "id") if c in reviews.columns), None)
        if identity_col:
            reviews["review_key"] = reviews[identity_col].astype(str)
        else:
            reviews["review_key"] = reviews.astype(str).agg("\x1f".join, axis=1).map(lambda value: hashlib.sha256(value.encode()).hexdigest())
        result_path = directory / "review_predictions.csv"
        embeddings_path = directory / "embeddings.npy"
        old = pd.read_csv(result_path) if not rebuild and result_path.is_file() else pd.DataFrame()
        seen = set(old.review_key.astype(str)) if "review_key" in old else set()
        new_reviews = reviews.loc[~reviews.review_key.astype(str).isin(seen)].copy()
        model = _model() if not new_reviews.empty or old.empty else None
        predictions, embedding_parts = [], []
        try:
            # Trained MLP / linear SVM are loaded from the package, never retrained here.
            supervised_classifiers = _supervised()[1]
        except FileNotFoundError:
            supervised_classifiers = {}
        for start in range(0, len(new_reviews), 1000):
            batch = new_reviews.iloc[start:start + 1000].copy()
            vectors = model.embed(batch.review_text.tolist()).astype(np.float32)
            scores, labels, margins, uncertain = model.classify_embeddings(vectors)
            batch["sentiment_label"] = labels
            batch["positive_similarity"], batch["negative_similarity"], batch["neutral_similarity"] = scores.T
            batch["classification_confidence"], batch["is_uncertain"] = margins, uncertain
            for classifier_name, classifier in supervised_classifiers.items():
                batch[f"{classifier_name}_label"] = classifier.predict(vectors)
            batch["embedding_row"] = np.arange(start, start + len(batch)) + (len(old) if not rebuild else 0)
            predictions.append(batch)
            embedding_parts.append(vectors.astype(np.float16))
        added = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        result = pd.concat(([old] if not old.empty and not rebuild else []) + ([added] if not added.empty else []), ignore_index=True)
        if result.empty:
            raise ValueError("No valid reviews are available for analysis.")
        result.to_csv(result_path, index=False)
        previous_vectors = np.load(embeddings_path).astype(np.float16, copy=False) if not rebuild and embeddings_path.is_file() else np.empty((0, 384), dtype=np.float16)
        new_vectors = np.concatenate(embedding_parts) if embedding_parts else np.empty((0, 384), dtype=np.float16)
        np.save(embeddings_path, np.concatenate((previous_vectors, new_vectors)))
        result, backfill_report = backfill_supervised_labels(result, result_path, embeddings_path, supervised_classifiers)
        model_dir = ROOT / "models" / "all-MiniLM-L6-v2"
        reference_path = ROOT / "models" / "reference_embeddings.npy"
        # The model is only loaded when new reviews need embedding; a pure resubmission has nothing to save.
        if model is not None and not model_dir.is_dir():
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(model_dir)
        if model is not None and not reference_path.is_file():
            model.save_reference_embeddings(reference_path)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Local SBERT model is not available: {exc}")
    summary = _summary(result)
    metadata = json.loads(record["metadata_json"])
    metadata.update({"reviews_retrieved": len(raw), "reviews_analyzed": len(result),
                     "review_status": status_label, "invalid_or_empty_reviews": int(len(raw) - len(result)),
                     "supervised_backfill": backfill_report})
    with database.connect() as db:
        metadata["uploaded_source"] = path.name
        db.execute("UPDATE analyses SET title=?,metadata_json=?,summary_json=?,fingerprint=?,reviews_path=?,embeddings_path=? WHERE id=? AND user_id=?",
                   (metadata.get("title") or "Untitled course", json.dumps(metadata), json.dumps(summary), fingerprint,
                    str(result_path), str(embeddings_path), analysis_id, user["id"]))
    _atomic_json(directory / "course_metadata.json", metadata)
    _atomic_json(directory / "course_summary.json", summary)
    _atomic_json(directory / "analysis_report.json", {"model": "all-MiniLM-L6-v2",
        "method": "reference embedding cosine similarity", "source_sha256": fingerprint,
        "reviews_analyzed": len(result), "new_reviews_analyzed": int(len(added)),
        "uncertain_review_count": summary["uncertain_review_count"], "evaluation": "manual sample only"})
    return {"id": analysis_id, "metadata": metadata, "summary": summary,
            "cache_reused": bool(len(new_reviews) == 0), "new_reviews_analyzed": int(len(added))}


@app.get("/courses")
def courses(user=Depends(_owner)):
    with database.connect() as db:
        rows = db.execute("SELECT * FROM analyses WHERE user_id=? ORDER BY created_at DESC", (user["id"],)).fetchall()
    return [_json_record(row) for row in rows]


@app.get("/analyses/history")
def history(user=Depends(_owner)):
    return courses(user)


@app.get("/courses/{analysis_id}")
def course(analysis_id: str, user=Depends(_owner)):
    return _json_record(_owned_analysis(analysis_id, user["id"]))


@app.get("/courses/{analysis_id}/analytics")
def analytics(analysis_id: str, user=Depends(_owner)):
    record = _owned_analysis(analysis_id, user["id"])
    summary = json.loads(record["summary_json"])
    with database.connect() as db:
        evaluation = db.execute("SELECT metrics_json FROM evaluations WHERE analysis_id=? AND user_id=?", (analysis_id, user["id"])).fetchone()
    evaluation_metrics = json.loads(evaluation["metrics_json"]) if evaluation else None
    result_path = Path(record["reviews_path"])
    if result_path.is_file() and result_path.name == "review_predictions.csv":
        frame = pd.read_csv(result_path)
        label_rows = frame.groupby("sentiment_label").size().to_dict()
        rating_rows = frame.assign(rating=pd.to_numeric(frame.rating, errors="coerce")).dropna(subset=["rating"]).groupby("rating").sentiment_label.value_counts().unstack(fill_value=0).to_dict(orient="index")
        return {"summary": summary, "sentiment_counts": label_rows, "rating_sentiment": rating_rows,
                "evaluation": evaluation_metrics,
                "predictions": frame.head(200).replace({np.nan: None}).to_dict("records")}
    return {"summary": summary, "predictions": [], "evaluation": evaluation_metrics}


@app.post("/courses/{analysis_id}/evaluation")
async def evaluate_sample(analysis_id: str, file: UploadFile = File(...), user=Depends(_owner)):
    _owned_analysis(analysis_id, user["id"])
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".json", ".jsonl", ".ndjson"}:
        raise HTTPException(415, "Upload a labelled CSV, JSON or JSON Lines file.")
    content = await file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, "Evaluation file exceeds the 20 MB limit.")
    try:
        if suffix == ".csv":
            frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
        elif suffix in {".jsonl", ".ndjson"} or content.lstrip().startswith(b"{") and b"\n{" in content:
            frame = pd.read_json(io.BytesIO(content), lines=True)
        else:
            frame = pd.read_json(io.BytesIO(content))
        text_col, label_col = evaluation_columns(frame)
        pairs = [(normalize_label(row[label_col]), clean_text(row[text_col])) for _, row in frame.iterrows()]
        pairs = [(label, text) for label, text in pairs if label and text]
        if not pairs:
            raise ValueError("No valid positive/negative/neutral labelled review rows were found.")
        analyzer = _model()
        truths, predictions, mistakes = [], [], []
        accumulator = EvaluationAccumulator()
        for start in range(0, len(pairs), 1000):
            batch = pairs[start:start + 1000]
            batch_truths = [truth for truth, _ in batch]
            _, batch_predictions, margins, _ = analyzer.classify([text for _, text in batch])
            accumulator.update(batch_truths, batch_predictions)
            for (truth, text), prediction, margin in zip(batch, batch_predictions, margins):
                if truth != prediction:
                    mistakes.append({"review_text": text[:600], "true_label": truth,
                                     "predicted_label": str(prediction), "classification_confidence": float(margin)})
        metrics = accumulator.report()
        metrics["misclassified_examples"] = mistakes[:100]
        directory = DATA / analysis_id
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content).hexdigest()
        (directory / f"evaluation_sample_{digest}{suffix}").write_bytes(content)
        _atomic_json(directory / f"evaluation_{digest}.json", metrics)
        with database.connect() as db:
            db.execute("INSERT INTO evaluations(analysis_id,user_id,metrics_json) VALUES(?,?,?) ON CONFLICT(analysis_id,user_id) DO UPDATE SET metrics_json=excluded.metrics_json",
                       (analysis_id, user["id"], json.dumps(metrics)))
        return {"evaluation": metrics, "source_fingerprint": digest,
                "note": "Measured only against the manually labelled sample supplied for this course."}
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400, f"Evaluation sample could not be evaluated: {exc}") from exc


@app.post("/courses/{analysis_id}/rebuild")
async def rebuild(analysis_id: str, file: UploadFile = File(...), user=Depends(_owner)):
    return await upload_reviews(analysis_id, file, True, user)


@app.delete("/courses/{analysis_id}")
def delete_course(analysis_id: str, user=Depends(_owner)):
    record = _owned_analysis(analysis_id, user["id"])
    with database.connect() as db:
        db.execute("DELETE FROM analyses WHERE id=? AND user_id=?", (analysis_id, user["id"]))
    import shutil
    shutil.rmtree(DATA / analysis_id, ignore_errors=True)
    return {"deleted": True}


@app.post("/courses/compare")
def compare(body: CompareBody, user=Depends(_owner)):
    placeholders = ",".join("?" for _ in body.course_ids)
    with database.connect() as db:
        rows = db.execute(f"SELECT * FROM analyses WHERE user_id=? AND id IN ({placeholders})", (user["id"], *body.course_ids)).fetchall()
    if len(rows) != len(set(body.course_ids)):
        raise HTTPException(404, "One or more analyses were not found for this account.")
    return [_json_record(row) for row in rows]


@app.post("/courses/{analysis_id}/ask")
def ask(analysis_id: str, body: AskBody, user=Depends(_owner)):
    record = _owned_analysis(analysis_id, user["id"])
    path = Path(record["reviews_path"])
    vectors_path = Path(record["embeddings_path"]) if record["embeddings_path"] else None
    if not path.is_file() or not vectors_path or not vectors_path.is_file():
        return {"answer": "There are no uploaded review embeddings for this course yet.", "evidence": []}
    try:
        model = _model()
        query = model.embed([body.question])[0]
        vectors = np.asarray(np.load(vectors_path, mmap_mode="r"), dtype=np.float32)
        norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        scores = (vectors / norms) @ query
        indices = np.argsort(scores)[-5:][::-1]
        frame = pd.read_csv(path)
        evidence = [{"review_text": str(frame.iloc[i].review_text)[:600], "sentiment": frame.iloc[i].sentiment_label,
                     "similarity": float(scores[i])} for i in indices]
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(500, "Could not retrieve saved review evidence.") from exc
    return {"answer": "Here are the most semantically relevant review excerpts. They are evidence to inspect; this local assistant does not generate unsupported claims.",
            "evidence": evidence}
