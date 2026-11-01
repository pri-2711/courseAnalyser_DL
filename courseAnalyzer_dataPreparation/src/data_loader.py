"""Streaming local input readers and review schema validation."""
from pathlib import Path
import pandas as pd

TEXT_FIELDS = ("review_clean", "review", "review_text", "text", "comment", "content")


def _field(columns, candidates):
    by_lower = {str(c).lower(): c for c in columns}
    return next((by_lower[c.lower()] for c in candidates if c.lower() in by_lower), None)


def iter_input(path, chunksize=5000):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, chunksize=chunksize, dtype=str, keep_default_na=False)
    elif suffix in (".jsonl", ".ndjson"):
        yield from pd.read_json(path, lines=True, chunksize=chunksize, dtype=False)
    elif suffix == ".json":
        # Recovered exports are newline-delimited JSON despite the .json extension.
        # Detect that form before falling back to a small JSON array/object load.
        with path.open("rb") as stream:
            first = stream.read(4096).lstrip()
        if first.startswith(b"{"):
            yield from pd.read_json(path, lines=True, chunksize=chunksize, dtype=False)
        else:
            frame = pd.read_json(path, dtype=False)
            for start in range(0, len(frame), chunksize):
                yield frame.iloc[start:start + chunksize].copy()
    else:
        raise ValueError(f"Unsupported input format {suffix!r}; use CSV, JSON, or JSONL")


def normalize_reviews(frame):
    text_col = _field(frame.columns, TEXT_FIELDS)
    if text_col is None:
        raise ValueError("No review text column found (review_clean, review, review_text, text, comment, content).")
    result = frame.copy()
    result["review_text"] = result[text_col].astype(str).str.strip()
    result = result[result.review_text != ""]
    platform_col = _field(result.columns, ("platform",))
    if platform_col:
        result["platform"] = result[platform_col].astype(str).str.strip().str.lower()
        result.loc[result.platform.isin(("", "nan", "none", "null")), "platform"] = "unknown"
    else:
        result["platform"] = "unknown"
    rating_col = _field(result.columns, ("rating", "score", "stars"))
    result["rating"] = pd.to_numeric(result[rating_col], errors="coerce") if rating_col else pd.NA
    date_col = _field(result.columns, ("review_date", "date", "created_at", "timestamp", "created"))
    result["review_date"] = pd.to_datetime(result[date_col], errors="coerce", utc=True) if date_col else pd.NaT
    return result
