"""Incremental review sentiment aggregates (not course-level scores)."""
LABELS = ("positive", "negative", "neutral")


def empty_counts():
    return {label: 0 for label in LABELS} | {"uncertain": 0, "total": 0}


def add_rows(counts, labels, uncertain):
    for label in LABELS:
        counts[label] += int((labels == label).sum())
    counts["uncertain"] += int(uncertain.sum())
    counts["total"] += len(labels)


def distribution(counts):
    total = counts["total"]
    return {"review_count": total,
            "sentiment_counts": {label: counts[label] for label in LABELS},
            "sentiment_ratios": {label: counts[label] / total if total else 0.0 for label in LABELS},
            "sentiment_value": (counts["positive"] - counts["negative"]) / total if total else 0.0,
            "average_rating": counts.get("rating_sum", 0.0) / counts["rating_n"] if counts.get("rating_n", 0) else None,
            "uncertain_review_count": counts["uncertain"]}
