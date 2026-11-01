"""Lightweight content-based recommendation over a user's own saved courses and resources.

Relevance is the cosine similarity between a query embedding and a resource embedding, both produced
by the existing local SBERT encoder (all-MiniLM-L6-v2). Embeddings are L2-normalised, so the dot
product is the cosine similarity. Resource embeddings are cached per analysis and recomputed only
when that resource's text changes.

Final score is a transparent weighted sum. A signal that a resource does not publish (no rating, no
reviews) contributes 0; it is never estimated or filled in.

    score = W_RELEVANCE * relevance
          + W_RATING    * (rating / 5)                      if a rating is published
          + W_SENTIMENT * ((sentiment_value + 1) / 2)       if reviews were analysed
"""
import hashlib
import json
from pathlib import Path

import numpy as np

W_RELEVANCE = 0.8
W_RATING = 0.1
W_SENTIMENT = 0.1
MIN_RELEVANCE = 0.2
TEXT_LIMIT = 1500
RANKING_WEIGHTS = {"relevance": W_RELEVANCE, "rating": W_RATING, "sentiment": W_SENTIMENT,
                   "minimum_relevance": MIN_RELEVANCE}


def resource_text(metadata):
    """Text that describes the resource: title, provider, description and the heading outline."""
    parts = [metadata.get("title"), metadata.get("provider"), metadata.get("description"),
             " ".join(metadata.get("learning_outline") or [])]
    return " ".join(str(part).strip() for part in parts if part and str(part).strip())[:TEXT_LIMIT].strip()


def cached_resource_vector(directory, text, encode):
    """Return the resource embedding, reusing the cached vector when the text is unchanged."""
    directory = Path(directory)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    vector_path, digest_path = directory / "resource_vector.npy", directory / "resource_vector.sha256"
    if vector_path.is_file() and digest_path.is_file() and digest_path.read_text(encoding="utf-8").strip() == digest:
        return np.load(vector_path)
    vector = np.asarray(encode([text])[0], dtype=np.float32)
    np.save(vector_path, vector)
    digest_path.write_text(digest, encoding="utf-8")
    return vector


def explain(relevance, rating, review_count, sentiment_value, reviews_analyzed):
    parts = [f"semantic similarity {relevance:.2f} to your query"]
    parts.append(f"rating {rating:.2f}/5" + (f" from {int(review_count):,} ratings" if review_count else "")
                 if rating is not None else "no rating published")
    if reviews_analyzed:
        parts.append(f"{reviews_analyzed:,} analysed reviews, sentiment value {sentiment_value:+.2f}")
    else:
        parts.append("no analysed reviews")
    return "; ".join(parts)


def rank(query_vector, resources, min_relevance=MIN_RELEVANCE):
    """resources: dicts with a unit-length `vector` plus the display fields listed below. Returns the
    resources at or above `min_relevance`, sorted by score (highest first)."""
    ranked = []
    query = np.asarray(query_vector, dtype=np.float32)
    for resource in resources:
        relevance = float(np.dot(query, resource["vector"]))
        if relevance < min_relevance:
            continue
        rating = resource.get("rating")
        sentiment = resource.get("sentiment_value") if resource.get("reviews_analyzed") else None
        score = W_RELEVANCE * relevance
        score += W_RATING * (float(rating) / 5.0) if rating is not None else 0.0
        score += W_SENTIMENT * ((float(sentiment) + 1.0) / 2.0) if sentiment is not None else 0.0
        ranked.append({
            **{key: resource[key] for key in ("id", "title", "provider", "platform", "url", "description",
                                              "resource_type", "rating", "review_count", "reviews_analyzed",
                                              "sentiment_counts", "sentiment_value")},
            "relevance": round(relevance, 4),
            "score": round(score, 4),
            "explanation": explain(relevance, rating, resource.get("review_count"), sentiment or 0.0,
                                   resource.get("reviews_analyzed") or 0),
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def metadata_from_row(row):
    return json.loads(row["metadata_json"] or "{}"), json.loads(row["summary_json"] or "{}")
