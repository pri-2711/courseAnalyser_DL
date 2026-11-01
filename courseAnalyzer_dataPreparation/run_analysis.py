"""Local, batched course review sentiment analysis and independent metadata analytics."""
import argparse
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from src.data_loader import iter_input, normalize_reviews
from src.evaluation import EvaluationAccumulator, evaluation_columns, normalize_label
from src.metadata import analyze_courses
from src.preprocessing import clean_text
from src.sbert_model import ReferenceSBERT
from src.scoring import add_rows, distribution, empty_counts, LABELS
from src.visualization import create_plots, create_course_plots

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("course-analyzer")
SUPPORTED = {".csv", ".json", ".jsonl", ".ndjson"}


def _discover_inputs():
    files = [p for p in (ROOT / "data/input").glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
    reviews = next((p for p in files if "review" in p.stem.lower()), None)
    courses = next((p for p in files if "course" in p.stem.lower() and "review" not in p.stem.lower() and "raw" not in p.stem.lower()), None)
    raw_courses = next((p for p in files if "course" in p.stem.lower() and "raw" in p.stem.lower()), None)
    evaluation = next((p for p in files if any(token in p.stem.lower() for token in ("manual", "label", "evaluation"))), None)
    if reviews is None:
        other = [p for p in files if p != courses and p != evaluation]
        if len(other) == 1:
            reviews = other[0]
    return reviews, courses, raw_courses, evaluation


def _signature(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prefix_signature(path, length):
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _jsonl_byte_offset(path, row_count):
    """Return the byte position after row_count newline-delimited JSON records."""
    with path.open("rb") as stream:
        for _ in range(row_count):
            if not stream.readline():
                break
        return stream.tell()


def _atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fingerprint_optional(path):
    return _signature(Path(path)) if path and Path(path).is_file() else None


def _archive_existing(output_dir, filename, version):
    source = output_dir / filename
    if not source.is_file():
        return
    history = output_dir / "history"
    history.mkdir(exist_ok=True)
    archive = history / f"{source.stem}_{version[:12]}{source.suffix}"
    if not archive.exists():
        import shutil
        shutil.copy2(source, archive)


def _load_latest(state_root):
    pointer = state_root / "latest.json"
    if not pointer.is_file():
        return None, None
    try:
        output_id = json.loads(pointer.read_text(encoding="utf-8"))["output_id"]
        if Path(output_id).name != output_id:
            return None, None
        output_dir = state_root / output_id
        manifest_path = output_dir / "manifest.json"
        if manifest_path.is_file():
            return output_dir, json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None, None
    return None, None


def _find_source_version(state_root, source_path):
    matches = []
    for folder in state_root.glob("analysis_*"):
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            item = json.loads(manifest_path.read_text(encoding="utf-8"))
            if item.get("source_path") == str(source_path):
                matches.append((int(item.get("source_bytes", 0)), folder, item))
        except (OSError, ValueError, TypeError):
            continue
    if not matches:
        return None, None
    _, folder, item = max(matches, key=lambda entry: entry[0])
    return folder, item


def _save_model_artifacts(analyzer, model_dir, reference_path):
    if model_dir.is_dir():
        if not reference_path.is_file() and hasattr(analyzer, "save_reference_embeddings"):
            analyzer.save_reference_embeddings(reference_path)
        return
    import uuid
    staging = model_dir.with_name(model_dir.name + ".tmp." + uuid.uuid4().hex)
    if hasattr(analyzer, "save_model"):
        analyzer.save_model(staging)
    else:
        staging.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        staging.replace(model_dir)
    if not reference_path.exists() and hasattr(analyzer, "save_reference_embeddings"):
        analyzer.save_reference_embeddings(reference_path)


def _rebuild_aggregates(predictions_path, output_dir):
    overall = empty_counts()
    platforms, ratings, trends = defaultdict(empty_counts), defaultdict(empty_counts), defaultdict(empty_counts)
    for frame in iter_input(predictions_path, 10000):
        # Chunked CSV readers keep a running index; positional lookups below need 0..n-1.
        frame = frame.reset_index(drop=True)
        labels = frame.sentiment_label.astype(str).to_numpy()
        uncertain = frame.is_uncertain.astype(str).str.lower().isin(("true", "1")).to_numpy()
        add_rows(overall, labels, uncertain)
        frame["_labels"], frame["_uncertain"] = labels, uncertain
        for name, group in frame.groupby("platform", sort=False):
            add_rows(platforms[str(name)], group["_labels"].to_numpy(), group["_uncertain"].to_numpy())
            rating = pd.to_numeric(group.rating, errors="coerce")
            valid = rating.notna()
            if valid.any():
                counts = platforms[str(name)]
                counts["rating_sum"] = counts.get("rating_sum", 0.0) + float(rating[valid].sum())
                counts["rating_n"] = counts.get("rating_n", 0) + int(valid.sum())
        ratings_numeric = pd.to_numeric(frame.rating, errors="coerce")
        valid = ratings_numeric.notna()
        if valid.any():
            overall["rating_sum"] = overall.get("rating_sum", 0.0) + float(ratings_numeric[valid].sum())
            overall["rating_n"] = overall.get("rating_n", 0) + int(valid.sum())
            for rating_value, indexes in frame.loc[valid].assign(_rating=ratings_numeric[valid].map(lambda x: str(float(x)))).groupby("_rating").groups.items():
                indexes = list(indexes)
                add_rows(ratings[str(rating_value)], labels[indexes], uncertain[indexes])
        if "review_date" in frame:
            dates = pd.to_datetime(frame.review_date, errors="coerce", utc=True)
            dated = frame.loc[dates.notna()].copy()
            if not dated.empty:
                dated["_day"] = dates[dates.notna()].dt.strftime("%Y-%m-%d")
                for (day, platform), indexes in dated.groupby(["_day", "platform"], sort=False).groups.items():
                    indexes = list(indexes)
                    add_rows(trends[(day, str(platform))], labels[indexes], uncertain[indexes])
    if not overall["total"]:
        raise ValueError("Persistent prediction file contains no valid review predictions.")
    summary = distribution(overall)
    platform_rows, rating_rows = _aggregate_rows(platforms, "platform"), _aggregate_rows(ratings, "rating")
    rating_rows.sort(key=lambda row: float(row["rating"]))
    trend_rows = []
    for (day, platform), counts in sorted(trends.items()):
        total = counts["total"]
        trend_rows.append({"date": day, "platform": platform, "review_count": total,
                           **{f"{label}_count": counts[label] for label in LABELS},
                           "sentiment_value": (counts["positive"] - counts["negative"]) / total if total else 0.0})
    pd.DataFrame([{"sentiment_label": label, "count": overall[label], "ratio": overall[label] / overall["total"],
                   "sentiment_value": summary["sentiment_value"]} for label in LABELS]).to_csv(output_dir / "sentiment_summary.csv", index=False)
    pd.DataFrame(platform_rows).to_csv(output_dir / "platform_sentiment.csv", index=False)
    pd.DataFrame(rating_rows).to_csv(output_dir / "rating_sentiment.csv", index=False)
    pd.DataFrame(trend_rows, columns=["date", "platform", "review_count", "positive_count", "negative_count", "neutral_count", "sentiment_value"]).to_csv(output_dir / "sentiment_trends.csv", index=False)
    for name, obj in (("sentiment_summary.json", summary), ("platform_sentiment.json", platform_rows),
                      ("rating_sentiment.json", rating_rows), ("sentiment_trends.json", trend_rows)):
        (output_dir / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    create_plots(summary, platform_rows, rating_rows, output_dir, trend_rows)
    return summary, platform_rows, rating_rows, trend_rows


def _renormalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _aggregate_rows(groups, key):
    rows = []
    for name, counts in sorted(groups.items(), key=lambda item: str(item[0])):
        total = counts["total"]
        rows.append({key: name, "review_count": total,
                     **{f"{label}_count": counts[label] for label in LABELS},
                     **{f"{label}_ratio": counts[label] / total if total else 0.0 for label in LABELS},
                     "sentiment_value": (counts["positive"] - counts["negative"]) / total if total else 0.0,
                     "average_rating": counts.get("rating_sum", 0.0) / counts.get("rating_n", 1) if counts.get("rating_n", 0) else None,
                     "uncertain_review_count": counts["uncertain"]})
    return rows


def _evaluate(path, analyzer, chunk_size, margin, output_dir):
    accumulator = EvaluationAccumulator()
    misclassified_path = output_dir / "misclassified_reviews.csv"
    wrote = False
    total = 0
    for raw in iter_input(path, chunk_size):
        text_col, label_col = evaluation_columns(raw)
        keep_text, truths, original_rows = [], [], []
        for _, row in raw.iterrows():
            truth, text = normalize_label(row[label_col]), clean_text(row[text_col])
            if truth and text:
                truths.append(truth); keep_text.append(text); original_rows.append(row.to_dict())
        if not keep_text:
            continue
        _, predicted, margins, _ = analyzer.classify(keep_text, margin)
        accumulator.update(truths, predicted)
        mistakes = []
        for source_row, truth, prediction, margin_value in zip(original_rows, truths, predicted, margins):
            if truth != prediction:
                mistakes.append({**source_row, "true_label": truth, "predicted_label": str(prediction),
                                 "classification_confidence": float(margin_value)})
        if mistakes:
            pd.DataFrame(mistakes).to_csv(misclassified_path, mode="a", index=False, header=not wrote)
            wrote = True
        total += len(keep_text)
    if not total:
        raise ValueError("Evaluation file has no valid manually labelled positive/negative/neutral reviews.")
    if not wrote:
        pd.DataFrame(columns=["true_label", "predicted_label", "classification_confidence"]).to_csv(misclassified_path, index=False)
    return accumulator.report()


def _load_course_file(path, output_dir, suffix=""):
    frames = list(iter_input(path, 10000))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary, data = analyze_courses(frame)
    metadata_columns = [c for c in ("course_title", "platform", "rating", "price", "subscribers", "reviews") if c in data]
    view = data[metadata_columns].copy() if metadata_columns else data.copy()
    view.to_csv(output_dir / f"course_analysis{suffix}.csv", index=False)
    if not view.empty:
        if "rating" in view and view.rating.notna().any():
            view.dropna(subset=["rating"]).nlargest(25, "rating").to_csv(output_dir / f"top_rated_courses{suffix}.csv", index=False)
        if "subscribers" in view and view.subscribers.notna().any():
            view.dropna(subset=["subscribers"]).nlargest(25, "subscribers").to_csv(output_dir / f"top_subscribed_courses{suffix}.csv", index=False)
    create_course_plots(data, output_dir, suffix)
    return summary


def run_analysis(input_path=None, output_base=None, courses_path=None, evaluation_path=None,
                 batch_size=64, chunk_size=5000, uncertainty_margin=0.03, device=None, analyzer=None,
                 courses_raw_path=None, rebuild=False):
    discovered_reviews, discovered_courses, discovered_raw_courses, discovered_evaluation = _discover_inputs()
    source = Path(input_path).expanduser().resolve() if input_path else (discovered_reviews.resolve() if discovered_reviews else None)
    if source is None or not source.is_file():
        raise FileNotFoundError("No review dataset found in data/input. Add a CSV/JSON/JSONL review file or pass --input.")
    if source.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported review input format: {source.suffix}")
    course_file = Path(courses_path).expanduser().resolve() if courses_path else discovered_courses
    raw_course_file = Path(courses_raw_path).expanduser().resolve() if courses_raw_path else discovered_raw_courses
    eval_file = Path(evaluation_path).expanduser().resolve() if evaluation_path else discovered_evaluation
    for label, path in (("course metadata", course_file), ("raw course metadata", raw_course_file), ("evaluation sample", eval_file)):
        if path and not Path(path).is_file():
            raise FileNotFoundError(f"Selected {label} file does not exist: {path}")
    review_hash, byte_size = _signature(source), source.stat().st_size
    course_hash, raw_hash, eval_hash = map(_fingerprint_optional, (course_file, raw_course_file, eval_file))
    state_root = Path(output_base).expanduser().resolve() if output_base else ROOT / "data/persisted"
    state_root.mkdir(parents=True, exist_ok=True)
    prior_dir, prior = _load_latest(state_root)
    if not prior or prior.get("source_path") != str(source):
        source_dir, source_manifest = _find_source_version(state_root, str(source))
        if source_manifest:
            prior_dir, prior = source_dir, source_manifest
    same_source = bool(prior and prior.get("source_path") == str(source))
    reviews_unchanged = bool(same_source and prior.get("source_sha256") == review_hash)
    same_classifier_config = bool(prior and prior.get("uncertainty_margin") == uncertainty_margin)
    can_append = bool(same_source and prior and byte_size > prior.get("source_bytes", 0)
                      and chunk_size == prior.get("chunk_size")
                      and same_classifier_config
                      and _prefix_signature(source, prior.get("source_bytes", 0)) == prior.get("source_prefix_sha256"))
    aux_unchanged = bool(prior and prior.get("courses_sha256") == course_hash
                         and prior.get("courses_raw_sha256") == raw_hash and prior.get("evaluation_sha256") == eval_hash)
    if not rebuild and reviews_unchanged and aux_unchanged and same_classifier_config and prior_dir and prior.get("inference_complete", True):
        _atomic_json(state_root / "latest.json", {"output_id": prior_dir.name})
        LOG.info("Inputs unchanged; loading persisted predictions and artifacts from %s", prior_dir)
        return prior_dir

    new_version = rebuild or not prior_dir or (not reviews_unchanged and not can_append) or (reviews_unchanged and not same_classifier_config)
    output_dir = state_root / datetime.now().strftime("analysis_%Y%m%d_%H%M%S_%f") if new_version else prior_dir
    output_dir.mkdir(parents=True, exist_ok=False) if new_version else None
    predictions_path = output_dir / "review_predictions.csv"
    embedding_dir = output_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models/all-MiniLM-L6-v2"
    reference_path = ROOT / "models/reference_embeddings.npy"
    do_inference = new_version or can_append
    aux_eval_changed = new_version or not prior or prior.get("evaluation_sha256") != eval_hash
    aux_courses_changed = new_version or not prior or prior.get("courses_sha256") != course_hash
    aux_raw_changed = new_version or not prior or prior.get("courses_raw_sha256") != raw_hash
    needs_model = do_inference or (eval_file and aux_eval_changed)
    if analyzer is None and needs_model:
        analyzer = ReferenceSBERT(batch_size=batch_size, device=device, model_dir=model_dir,
                                  reference_embeddings_path=reference_path,
                                  require_local=bool(prior_dir and prior and prior.get("inference_complete", True)))

    rows_processed = 0 if new_version else int(prior.get("rows_processed", 0))
    next_embedding_chunk = 0 if new_version else int(prior.get("next_embedding_chunk", 0))
    skipped = 0 if new_version else int(prior.get("empty_reviews_skipped", 0))
    if do_inference:
        source_columns = None
        rows_before = 0
        for raw in iter_input(source, chunk_size):
            if source_columns is None:
                source_columns = list(raw.columns)
                if can_append and prior.get("input_columns") and source_columns != prior["input_columns"]:
                    raise ValueError("Review columns changed while appending; use --rebuild to start a clean dataset version.")
            offset = 0 if new_version else int(prior.get("rows_processed", 0))
            begin = max(0, offset - rows_before)
            rows_before += len(raw)
            new_raw = raw.iloc[begin:]
            if new_raw.empty:
                continue
            rows_processed += len(new_raw)
            frame = normalize_reviews(new_raw)
            texts = [clean_text(value) for value in frame.review_text.tolist()]
            valid = [i for i, value in enumerate(texts) if value]
            skipped += len(new_raw) - len(valid)
            frame = frame.iloc[valid].copy().reset_index(drop=True)
            texts = [texts[i] for i in valid]
            chunk_id = next_embedding_chunk
            next_embedding_chunk += 1
            cache_file = embedding_dir / f"chunk_{chunk_id:06d}.npy"
            if not texts:
                np.save(cache_file, np.empty((0, 384), dtype="float16"))
                continue
            vectors = _renormalize(analyzer.embed(texts))
            temporary_cache = cache_file.with_name(cache_file.stem + ".tmp.npy")
            np.save(temporary_cache, vectors.astype("float16"))
            temporary_cache.replace(cache_file)
            vectors = _renormalize(np.load(cache_file, mmap_mode="r"))
            scores, labels, margins, uncertain = analyzer.classify_embeddings(vectors, uncertainty_margin)
            frame["review_text"], frame["embedding_chunk"] = texts, chunk_id
            frame["sentiment_label"] = labels
            frame["positive_similarity"], frame["negative_similarity"], frame["neutral_similarity"] = scores.T
            frame["classification_confidence"], frame["is_uncertain"] = margins, uncertain
            frame.to_csv(predictions_path, mode="a", index=False, header=not predictions_path.exists(), encoding="utf-8")
            LOG.info("Generated predictions for %s new reviews", len(frame))
            is_jsonl = source.suffix.lower() in {".jsonl", ".ndjson"}
            if source.suffix.lower() == ".json":
                with source.open("rb") as source_stream:
                    is_jsonl = source_stream.read(1) == b"{"
            if is_jsonl:
                checkpoint_bytes = _jsonl_byte_offset(source, rows_processed)
                checkpoint = {"source_path": str(source), "source_sha256": review_hash,
                              "source_bytes": checkpoint_bytes,
                              "source_prefix_sha256": _prefix_signature(source, checkpoint_bytes),
                              "rows_processed": rows_processed, "empty_reviews_skipped": skipped,
                              "next_embedding_chunk": next_embedding_chunk, "chunk_size": chunk_size,
                              "input_columns": source_columns, "uncertainty_margin": uncertainty_margin,
                              "courses_sha256": None, "courses_raw_sha256": None,
                              "evaluation_sha256": None, "course_metadata": {},
                              "model": "all-MiniLM-L6-v2", "output_id": output_dir.name,
                              "inference_complete": False}
                _atomic_json(output_dir / "manifest.json", checkpoint)
                _atomic_json(state_root / "latest.json", {"output_id": output_dir.name})
        if new_version and not predictions_path.exists():
            raise ValueError("No non-empty review texts were found in the selected input.")

    if predictions_path.exists():
        summary, platform_rows, rating_rows, trend_rows = _rebuild_aggregates(predictions_path, output_dir)
    elif prior_dir and prior:
        # Metadata/evaluation-only refreshes keep the persisted review results intact.
        summary = json.loads((output_dir / "sentiment_summary.json").read_text(encoding="utf-8"))
        platform_rows = json.loads((output_dir / "platform_sentiment.json").read_text(encoding="utf-8"))
        rating_rows = json.loads((output_dir / "rating_sentiment.json").read_text(encoding="utf-8"))
        trend_rows = json.loads((output_dir / "sentiment_trends.json").read_text(encoding="utf-8"))
    else:
        raise ValueError("No persistent review predictions are available.")

    if aux_eval_changed:
        if not new_version:
            _archive_existing(output_dir, "evaluation_metrics.json", prior.get("evaluation_sha256") or "no-eval")
            _archive_existing(output_dir, "misclassified_reviews.csv", prior.get("evaluation_sha256") or "no-eval")
        if eval_file:
            evaluation = _evaluate(eval_file, analyzer, chunk_size, uncertainty_margin, output_dir)
            eval_note = "Metrics are measured only against the supplied manually labelled sample."
        else:
            evaluation = None
            eval_note = "Not computed: provide an independent manually labelled sample using --evaluation."
        (output_dir / "evaluation_metrics.json").write_text(json.dumps({"evaluation": evaluation, "note": eval_note}, indent=2), encoding="utf-8")
    course_summary = prior.get("course_metadata", {}) if prior and not aux_courses_changed and not aux_raw_changed else {}
    if aux_courses_changed:
        if not new_version:
            _archive_existing(output_dir, "course_analysis.csv", prior.get("courses_sha256") or "no-courses")
        if course_file and Path(course_file).is_file():
            course_summary["courses"] = _load_course_file(course_file, output_dir)
        else:
            course_summary.pop("courses", None)
    if aux_raw_changed:
        if not new_version:
            _archive_existing(output_dir, "course_analysis_raw.csv", prior.get("courses_raw_sha256") or "no-raw-courses")
        if raw_course_file and Path(raw_course_file).is_file():
            course_summary["courses_raw"] = _load_course_file(raw_course_file, output_dir, "_raw")
        else:
            course_summary.pop("courses_raw", None)
    report = {"input": str(source), "reviews_analyzed": summary["review_count"], "empty_reviews_skipped": skipped,
              "model": "all-MiniLM-L6-v2", "method": "reference embedding cosine similarity",
              "embedding_cache_directory": str(embedding_dir), "embedding_chunks_persisted": next_embedding_chunk,
              "model_directory": str(model_dir), "reference_embeddings_path": str(reference_path),
              "platforms": len(platform_rows), "ratings_with_numeric_values": len(rating_rows),
              "course_metadata": course_summary or None, "dated_platform_groups": len(trend_rows),
              "review_course_association": "not performed", "evaluation_sample": str(eval_file) if eval_file else None,
              "review_fingerprint": review_hash, "uncertainty_margin": uncertainty_margin,
              "rebuild": bool(rebuild)}
    (output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if analyzer and needs_model:
        _save_model_artifacts(analyzer, model_dir, reference_path)
    source_info = {"source_path": str(source), "source_sha256": review_hash, "source_bytes": byte_size,
                   "source_prefix_sha256": _prefix_signature(source, byte_size), "rows_processed": rows_processed,
                   "empty_reviews_skipped": skipped, "next_embedding_chunk": next_embedding_chunk,
                   "chunk_size": chunk_size, "input_columns": source_columns if do_inference else (prior or {}).get("input_columns"),
                   "uncertainty_margin": uncertainty_margin,
                   "courses_sha256": course_hash, "courses_raw_sha256": raw_hash, "evaluation_sha256": eval_hash,
                   "model": "all-MiniLM-L6-v2", "output_id": output_dir.name, "inference_complete": True,
                   "course_metadata": course_summary}
    _atomic_json(output_dir / "manifest.json", source_info)
    _atomic_json(state_root / "latest.json", {"output_id": output_dir.name})
    LOG.info("Loaded/generated persistent analysis at %s", output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Review CSV, JSON, or JSONL; auto-detected in data/input when omitted")
    parser.add_argument("--courses", default=None, help="Optional independent course metadata file")
    parser.add_argument("--courses-raw", default=None, help="Optional separate raw course catalog export")
    parser.add_argument("--evaluation", default=None, help="Optional manually labelled evaluation sample")
    parser.add_argument("--output-dir", default=None, help="Persistent analysis state directory (default: data/persisted)")
    parser.add_argument("--rebuild", action="store_true", help="Recompute all review embeddings and predictions as a new persistent version")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--uncertainty-margin", type=float, default=0.03)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_analysis(args.input, args.output_dir, args.courses, args.evaluation, args.batch_size,
                     args.chunk_size, args.uncertainty_margin, args.device, courses_raw_path=args.courses_raw,
                     rebuild=args.rebuild)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
