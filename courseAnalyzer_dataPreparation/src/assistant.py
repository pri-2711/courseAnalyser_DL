"""Local semantic retrieval that answers with retrieved review evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data_loader import iter_input


def retrieve_reviews(question, analyzer, predictions_path, embedding_dir, top_k=6, sentiment_filter=None):
    query = analyzer.embed([question])[0]
    best = []
    pending_id, pending = None, []

    def score_group(chunk_id, group):
        cache_file = Path(embedding_dir) / f"chunk_{int(chunk_id):06d}.npy"
        if not cache_file.exists():
            return
        vectors = np.asarray(np.load(cache_file, mmap_mode="r"), dtype=np.float32)
        if len(vectors) != len(group):
            return
        if sentiment_filter:
            selected = np.flatnonzero(group.sentiment_label.to_numpy() == sentiment_filter)
            if not len(selected):
                return
            group = group.iloc[selected]
            vectors = vectors[selected]
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        similarities = vectors @ query
        take = min(top_k, len(similarities))
        for idx in np.argpartition(similarities, -take)[-take:]:
            best.append((float(similarities[idx]), group.iloc[idx].to_dict()))
        best.sort(key=lambda x: x[0], reverse=True)
        del best[top_k:]

    for pred in iter_input(predictions_path, 5000):
        for chunk_id, group in pred.groupby("embedding_chunk", sort=False):
            if pending_id is None:
                pending_id = chunk_id
            if chunk_id != pending_id:
                score_group(pending_id, pd.concat(pending, ignore_index=True))
                pending_id, pending = chunk_id, []
            pending.append(group.copy())
    if pending:
        score_group(pending_id, pd.concat(pending, ignore_index=True))
    return best


def evidence_answer(question, matches):
    if not matches:
        return "No cached review embeddings were found. Run the review analysis first."
    excerpts = [str(row.get("review_text", ""))[:500] for _, row in matches]
    return ("Here are the most semantically relevant review excerpts. Treat these as evidence to inspect, "
            "not as a statistically validated summary:\n\n" + "\n\n".join(f"- {text}" for text in excerpts))


def answer_question(question, analyzer, predictions_path, embedding_dir, summary, platform_rows, courses, top_k=6):
    q = question.lower()
    if "overall sentiment" in q or "sentiment overall" in q:
        counts = summary.get("sentiment_counts", {})
        return (f"Across {summary.get('review_count', 0):,} analyzed reviews, predicted sentiment counts are "
                f"{counts.get('positive', 0):,} positive, {counts.get('negative', 0):,} negative, and "
                f"{counts.get('neutral', 0):,} neutral. Sentiment value is {summary.get('sentiment_value', 0):.3f}."), []
    if "platform" in q and any(word in q for word in ("positive", "sentiment", "most satisfied")) and platform_rows:
        best = max(platform_rows, key=lambda row: row.get("positive_ratio", 0))
        return (f"{best['platform']} has the largest positive review share in the available platform aggregates "
                f"({best['positive_ratio']:.1%}; {best['review_count']:,} reviews)."), []
    if any(term in q for term in ("highest-rated", "highest rated", "top-rated", "top rated")):
        field = "rating"
        if not courses.empty and field in courses and "course_title" in courses:
            rows = courses.dropna(subset=[field]).nlargest(top_k, field)
            return "Highest course ratings in the separate metadata catalog:", rows.to_dict("records")
        return "Course rating metadata is unavailable in this run.", []
    if any(term in q for term in ("subscribers", "most popular", "most subscribed")):
        if not courses.empty and "subscribers" in courses and "course_title" in courses:
            rows = courses.dropna(subset=["subscribers"]).nlargest(top_k, "subscribers")
            return "Largest subscriber counts in the separate metadata catalog:", rows.to_dict("records")
        return "Subscriber metadata is unavailable in this run.", []
    label_filter = "negative" if any(term in q for term in ("complaint", "complain", "negative", "problem", "issue")) else None
    if any(term in q for term in ("satisfied", "satisfaction", "positive")):
        label_filter = "positive"
    matches = retrieve_reviews(question, analyzer, predictions_path, embedding_dir, top_k, label_filter)
    return evidence_answer(question, matches), matches
