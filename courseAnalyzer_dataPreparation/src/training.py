"""Supervised sentiment training on rating-derived weak labels."""
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from .data_loader import _field, iter_input
from .mlp import MLP_LABELS, MLPPredictor, model_size_bytes, save_mlp, train_mlp
from .preprocessing import clean_text
from .sbert_model import ReferenceSBERT

LOG = logging.getLogger("course-analyzer.training")
LABELS = ("negative", "neutral", "positive")
MODEL_CATEGORIES = {
    "sbert_reference_baseline": "Transformer feature extractor (SBERT cosine-similarity baseline, no supervised fitting)",
    "logistic_regression": "Traditional Machine Learning",
    "linear_svm": "Traditional Machine Learning",
    "mlp": "Deep Learning (supervised PyTorch MLP on SBERT embeddings)",
}
BASELINE_LABELS = ("positive", "negative", "neutral")
LABEL_MAPPING = {"negative": [1, 2], "neutral": [3], "positive": [4, 5]}
SEED = 42
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
PREPROCESSING = {
    "text_field": "review_clean (or recognized review text field)",
    "text_cleanup": "HTML removal, HTML entity decoding, whitespace collapse, strip",
    "duplicate_key": "SHA-256 of cleaned, case-folded, whitespace-normalized review text",
    "rating_labels": LABEL_MAPPING,
}
TRAINING_DIRNAME = "trained_sentiment"


def rating_to_label(value):
    """Map integer star ratings to weak sentiment labels; reject missing/invalid values."""
    try:
        rating = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(rating):
        return None
    if rating in (1, 2):
        return "negative"
    if rating == 3:
        return "neutral"
    if rating in (4, 5):
        return "positive"
    return None


def normalized_text_key(text):
    normalized = " ".join(clean_text(text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prefix_sha256(path, size):
    digest = hashlib.sha256()
    remaining = size
    with Path(path).open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _rating_value(value):
    if value is None:
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return rating if np.isfinite(rating) else None


def load_weak_labeled_dataset(path, chunk_size=20000):
    """Audit the file and collapse duplicate text to its unambiguous majority label."""
    path = Path(path).resolve()
    fingerprint = _file_sha256(path)
    groups = {}
    audit = {
        "source": str(path),
        "dataset_fingerprint_sha256": fingerprint,
        "row_count": 0,
        "columns": [],
        "rating_distribution": {},
        "rating_missing_count": 0,
        "rating_non_numeric_count": 0,
        "rating_out_of_range_count": 0,
        "rating_in_range_unmapped_count": 0,
        "valid_rating_count": 0,
        "text_missing_count": 0,
        "text_empty_after_cleaning_count": 0,
        "text_length_mean": 0.0,
        "text_length_median": 0.0,
        "text_length_at_most_20_count": 0,
        "text_length_over_5000_count": 0,
        "duplicate_text_rows_after_first": 0,
        "unique_nonempty_normalized_texts": 0,
        "weak_label_counts_before_deduplication": {label: 0 for label in LABELS},
        "existing_sentiment_label_columns": [],
        "course_or_user_group_fields": [],
        "rating_label_definition": LABEL_MAPPING,
        "weak_label_caveat": "Labels are derived from ratings and are not verified sentiment ground truth.",
    }
    lengths = []
    rating_counts = Counter()
    row_index = 0
    text_fields = ("review_clean", "review", "review_text", "text", "comment", "content")
    rating_fields = ("rating", "score", "stars")
    user_course_fields = {"course_id", "courseid", "course_identifier", "user_id", "userid",
                          "user_identifier", "student_id", "author_id"}

    for frame in iter_input(path, chunk_size):
        if not audit["columns"]:
            audit["columns"] = [str(column) for column in frame.columns]
            audit["existing_sentiment_label_columns"] = [
                column for column in frame.columns if str(column).lower() in {"label", "sentiment", "sentiment_label"}
            ]
            audit["course_or_user_group_fields"] = [
                column for column in frame.columns if str(column).lower() in user_course_fields
            ]
        text_col = _field(frame.columns, text_fields)
        rating_col = _field(frame.columns, rating_fields)
        if text_col is None:
            raise ValueError(f"No supported review text column in {path.name}.")
        if rating_col is None:
            raise ValueError(f"No rating field found; expected one of {rating_fields}.")

        texts = frame[text_col].tolist()
        ratings = frame[rating_col].tolist()
        audit["row_count"] += len(frame)
        for text_value, rating_value in zip(texts, ratings):
            rating = _rating_value(rating_value)
            text_missing = text_value is None or pd.isna(text_value)
            text = "" if text_missing else clean_text(text_value)
            if text_missing:
                audit["text_missing_count"] += 1
            if not text:
                audit["text_empty_after_cleaning_count"] += 1
            lengths.append(len(text))
            if len(text) <= 20:
                audit["text_length_at_most_20_count"] += 1
            if len(text) > 5000:
                audit["text_length_over_5000_count"] += 1

            if rating is None:
                raw = "" if text_missing else str(rating_value).strip()
                if raw.lower() in {"", "nan", "none", "null"}:
                    audit["rating_missing_count"] += 1
                else:
                    audit["rating_non_numeric_count"] += 1
            else:
                rating_key = str(int(rating)) if rating.is_integer() else str(rating)
                rating_counts[rating_key] += 1
                if not 1 <= rating <= 5:
                    audit["rating_out_of_range_count"] += 1
                else:
                    audit["valid_rating_count"] += 1
            label = rating_to_label(rating)
            if rating is not None and 1 <= rating <= 5 and label is None:
                audit["rating_in_range_unmapped_count"] += 1
            if label and text:
                audit["weak_label_counts_before_deduplication"][label] += 1
            key = normalized_text_key(text)
            if key:
                group = groups.get(key)
                if group is None:
                    group = groups[key] = {"text": text, "labels": Counter(), "first_row": row_index}
                else:
                    audit["duplicate_text_rows_after_first"] += 1
                if label:
                    group["labels"][label] += 1
            row_index += 1

    audit["unique_nonempty_normalized_texts"] = len(groups)
    audit["rating_distribution"] = dict(sorted(rating_counts.items(), key=lambda item: float(item[0])))
    if lengths:
        audit["text_length_mean"] = round(float(np.mean(lengths)), 2)
        audit["text_length_median"] = round(float(np.median(lengths)), 2)
    conflicting_groups = 0
    tied_groups = 0
    examples = []
    for key, group in groups.items():
        counts = group["labels"]
        if not counts:
            continue
        if len(counts) > 1:
            conflicting_groups += 1
        largest = max(counts.values())
        winners = [label for label, count in counts.items() if count == largest]
        if len(winners) != 1:
            tied_groups += 1
            continue
        examples.append((group["text"], winners[0], key, group["first_row"]))
    audit["duplicate_text_groups_with_conflicting_weak_labels"] = conflicting_groups
    audit["ambiguous_tied_label_groups_excluded_from_training"] = tied_groups
    audit["unique_weak_labeled_examples"] = len(examples)
    audit["weak_label_counts_after_duplicate_collapse"] = dict(Counter(item[1] for item in examples))
    audit["duplicate_handling"] = (
        "Use one example per normalized review text; majority rating-derived label wins. "
        "Text groups tied across labels are excluded from supervised fitting, and remain untouched in the source."
    )
    texts = [item[0] for item in examples]
    labels = np.asarray([item[1] for item in examples], dtype=str)
    group_ids = np.asarray([item[2] for item in examples], dtype=str)
    first_rows = np.asarray([item[3] for item in examples], dtype=np.int64)
    if len(set(labels)) < 3:
        raise ValueError("Training requires at least one unambiguous example for each weak-label class.")
    return audit, texts, labels, group_ids, first_rows


def stratified_group_splits(labels, group_ids, seed=SEED):
    """Return duplicate-safe 64/16/20 approximately stratified train/validation/test indices."""
    labels = np.asarray(labels)
    group_ids = np.asarray(group_ids)
    indices = np.arange(len(labels))
    first = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_validation, test = next(first.split(indices, labels, groups=group_ids))
    second = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_relative, validation_relative = next(
        second.split(train_validation, labels[train_validation], groups=group_ids[train_validation])
    )
    train = train_validation[train_relative]
    validation = train_validation[validation_relative]
    splits = {"train": train, "validation": validation, "test": test}
    for left_name, left in splits.items():
        for right_name, right in splits.items():
            if left_name < right_name and set(group_ids[left]).intersection(group_ids[right]):
                raise RuntimeError(f"Duplicate review leakage between {left_name} and {right_name}.")
    if any(set(labels[idx]) != set(LABELS) for idx in splits.values()):
        raise ValueError("A stratified split is missing one or more sentiment classes.")
    return splits


def _resolve_encoder(model_root, batch_size):
    model_root = Path(model_root)
    local_model = model_root / "all-MiniLM-L6-v2"
    if local_model.is_dir():
        return ReferenceSBERT(model_dir=local_model, batch_size=batch_size, require_local=True)
    cache_root = Path(os.environ.get(
        "HF_HUB_CACHE",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
    ))
    repository = cache_root / "models--sentence-transformers--all-MiniLM-L6-v2"
    reference = repository / "refs" / "main"
    revision = reference.read_text(encoding="utf-8").strip() if reference.is_file() else ""
    snapshot = repository / "snapshots" / revision
    if snapshot.is_dir():
        return ReferenceSBERT(model_dir=snapshot, batch_size=batch_size, require_local=True)
    return ReferenceSBERT(model_name=MODEL_ID, batch_size=batch_size)


def _valid_legacy_embedding_cache(root, source, fingerprint):
    latest = root / "data" / "persisted" / "latest.json"
    if not latest.is_file():
        return None
    try:
        output_id = json.loads(latest.read_text(encoding="utf-8"))["output_id"]
        if Path(output_id).name != output_id:
            return None
        output = root / "data" / "persisted" / output_id
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if Path(manifest.get("source_path", "")).resolve() != Path(source).resolve():
        return None
    if manifest.get("source_sha256") != fingerprint or manifest.get("empty_reviews_skipped") != 0:
        return None
    prefix_size = int(manifest.get("source_bytes", 0))
    if prefix_size <= 0 or _prefix_sha256(source, prefix_size) != manifest.get("source_prefix_sha256"):
        return None
    predictions = output / "review_predictions.csv"
    chunk_size = int(manifest.get("chunk_size", 0))
    if not predictions.is_file() or chunk_size <= 0:
        return None
    return {"output": output, "predictions": predictions, "rows": int(manifest.get("rows_processed", 0)),
            "chunk_size": chunk_size}


def _reuse_legacy_embeddings(cache, texts, group_ids, first_rows, legacy, progress):
    if legacy is None or not len(texts):
        return 0
    first_row_map = {int(row): i for i, row in enumerate(first_rows)}
    reused = 0
    row = 0
    try:
        reader = pd.read_csv(legacy["predictions"], usecols=["review_text", "embedding_chunk"],
                             chunksize=legacy["chunk_size"], dtype=str, keep_default_na=False)
        for frame in reader:
            chunk_id = int(frame["embedding_chunk"].iloc[0]) if not frame.empty else row // legacy["chunk_size"]
            if not frame.empty and not (frame["embedding_chunk"] == str(chunk_id)).all():
                return 0
            chunk_path = legacy["output"] / "embeddings" / f"chunk_{chunk_id:06d}.npy"
            if not chunk_path.is_file():
                return 0
            vectors = np.load(chunk_path, mmap_mode="r")
            if len(vectors) != len(frame):
                return 0
            for offset, review_text in enumerate(frame["review_text"].tolist()):
                source_row = row + offset
                example_idx = first_row_map.get(source_row)
                if example_idx is None:
                    continue
                if normalized_text_key(review_text) != group_ids[example_idx]:
                    return 0
                cache[example_idx] = vectors[offset]
                reused += 1
            row += len(frame)
            if row >= legacy["rows"]:
                break
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return 0
    if row < legacy["rows"]:
        return 0
    progress({"stage": "embedding", "progress": 0.28, "message": f"Reused {reused:,} verified cached review embeddings."})
    return reused


def _ensure_embeddings(root, fingerprint, texts, first_rows, encoder, progress):
    cache_dir = Path(root) / "data" / "training_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{fingerprint}_{hashlib.sha256(MODEL_ID.encode()).hexdigest()[:12]}.npy"
    if cache_path.is_file():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == (len(texts), 384) and cached.dtype == np.float16:
            progress({"stage": "embedding", "progress": 0.35, "message": "Reusing cached SBERT embeddings."})
            return cache_path, True
    temporary_path = cache_path.with_name(cache_path.stem + ".tmp.npy")
    target = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=np.float16, shape=(len(texts), 384))
    legacy = _valid_legacy_embedding_cache(Path(root), Path(root) / "data" / "input" / "reviews.json", fingerprint)
    group_ids = [normalized_text_key(text) for text in texts]
    reused = _reuse_legacy_embeddings(target, texts, group_ids, first_rows, legacy, progress)
    filled = np.zeros(len(texts), dtype=bool)
    if reused:
        # A first-row index can be reused only when its matching legacy prediction was verified.
        rows = set(first_rows[first_rows < legacy["rows"]].tolist())
        for idx, source_row in enumerate(first_rows):
            if int(source_row) in rows:
                filled[idx] = True
    batch_size = 512
    missing_indices = np.flatnonzero(~filled)
    for start in range(0, len(missing_indices), batch_size):
        batch_indices = missing_indices[start:start + batch_size]
        vectors = encoder.embed([texts[i] for i in batch_indices])
        target[batch_indices] = np.asarray(vectors, dtype=np.float16)
        progress({"stage": "embedding", "progress": 0.28 + 0.42 * min(1.0, (start + len(batch_indices)) / max(1, len(missing_indices))),
                  "message": f"Encoded {min(start + len(batch_indices), len(missing_indices)):,} of {len(missing_indices):,} uncached examples."})
    target.flush()
    del target
    temporary_path.replace(cache_path)
    return cache_path, False


def _metric_report(y_true, y_pred, scores=None, score_labels=None):
    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        score_labels = tuple(score_labels or LABELS)
        reordered = np.column_stack([scores[:, score_labels.index(label)] for label in LABELS])
        one_vs_rest = {}
        for position, label in enumerate(LABELS):
            try:
                one_vs_rest[label] = float(roc_auc_score((np.asarray(y_true) == label).astype(int), reordered[:, position]))
            except ValueError:
                one_vs_rest[label] = None
        try:
            report["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, reordered, labels=list(LABELS),
                                                                multi_class="ovr", average="macro"))
        except ValueError:
            report["roc_auc_ovr_macro"] = None
        report["roc_auc_ovr_per_class"] = one_vs_rest
    return report


def _predict_batched(model, features, indices, batch_size=8192):
    outputs = []
    started = time.perf_counter()
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        outputs.extend(model.predict(np.asarray(features[batch_idx], dtype=np.float32)).tolist())
    return np.asarray(outputs), time.perf_counter() - started


def _scores_batched(model, features, indices, batch_size=8192):
    """Return class-aligned probabilities or decision scores for ROC-AUC reporting."""
    output = []
    for start in range(0, len(indices), batch_size):
        batch = np.asarray(features[indices[start:start + batch_size]], dtype=np.float32)
        if hasattr(model, "predict_proba"):
            output.append(model.predict_proba(batch))
        else:
            output.append(model.decision_function(batch))
    return np.vstack(output), tuple(str(label) for label in model.classes_)


def build_classifiers(seed=SEED):
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=500, solver="lbfgs", random_state=seed,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced", dual="auto", max_iter=5000, random_state=seed,
        ),
    }


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _sklearn_size(model):
    import joblib
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.tell()


def save_selected_classifier(classifier, training_dir, metadata):
    training_dir = Path(training_dir)
    training_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    model_path = training_dir / "classifier.joblib"
    temporary_model = training_dir / "classifier.joblib.tmp"
    joblib.dump(classifier, temporary_model)
    temporary_model.replace(model_path)
    _write_json(training_dir / "metadata.json", metadata)
    return model_path


def load_saved_classifier(training_dir):
    """Load the selected supervised classifier and its metadata, without training."""
    training_dir = Path(training_dir)
    model_path = training_dir / "classifier.joblib"
    metadata_path = training_dir / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        return None
    import joblib
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return joblib.load(model_path), metadata


def run_training(input_path, project_root, progress=None, seed=SEED, embedding_batch_size=64):
    """Train and evaluate baseline, logistic regression, and linear SVM."""
    root = Path(project_root).resolve()
    source = Path(input_path).resolve()
    progress = progress or (lambda _state: None)
    progress({"stage": "auditing", "progress": 0.02, "message": "Auditing rating-derived labels and duplicate reviews."})
    audit, texts, labels, group_ids, first_rows = load_weak_labeled_dataset(source)
    if len(texts) < 30:
        raise ValueError("At least 30 unambiguous unique review texts are required for grouped stratified splits.")
    progress({"stage": "splitting", "progress": 0.12, "message": f"Prepared {len(texts):,} unique weak-labeled texts."})
    splits = stratified_group_splits(labels, group_ids, seed)
    split_info = {
        "seed": seed,
        "method": "two-stage StratifiedGroupKFold(n_splits=5, shuffle=True)",
        "group_key": "normalized review text SHA-256",
        "sizes": {name: int(len(indices)) for name, indices in splits.items()},
        "fractions": {name: round(len(indices) / len(labels), 5) for name, indices in splits.items()},
        "class_counts": {name: dict(Counter(labels[indices])) for name, indices in splits.items()},
    }
    training_dir = root / "models" / TRAINING_DIRNAME
    training_dir.mkdir(parents=True, exist_ok=True)
    _write_json(training_dir / "dataset_audit.json", audit)
    split_frame = pd.DataFrame({"review_text_sha256": group_ids, "weak_label": labels, "split": ""})
    for name, indices in splits.items():
        split_frame.loc[indices, "split"] = name
    split_frame.to_csv(training_dir / "split_assignments.csv", index=False)
    encoder = _resolve_encoder(root / "models", embedding_batch_size)
    references = tuple(np.asarray(refs, dtype=np.float32) for refs in encoder.references)
    cache_path, reused_cache = _ensure_embeddings(root, audit["dataset_fingerprint_sha256"], texts,
                                                   first_rows, encoder, progress)
    model_version = importlib.metadata.version("sentence-transformers")
    del encoder
    gc.collect()
    features = np.load(cache_path, mmap_mode="r")
    train_indices = splits["train"]
    validation_indices = splits["validation"]
    test_indices = splits["test"]

    baseline_predictions, baseline_scores = [], []
    baseline_started = time.perf_counter()
    for start in range(0, len(test_indices), 8192):
        batch_indices = test_indices[start:start + 8192]
        batch_vectors = np.asarray(features[batch_indices], dtype=np.float32)
        scores = np.column_stack([np.max(batch_vectors @ refs.T, axis=1) for refs in references])
        baseline_predictions.extend(np.asarray(BASELINE_LABELS)[scores.argmax(axis=1)].tolist())
        baseline_scores.append(scores)
    baseline_inference_seconds = time.perf_counter() - baseline_started
    baseline_report = _metric_report(labels[test_indices], np.asarray(baseline_predictions),
                                     np.vstack(baseline_scores), BASELINE_LABELS)

    cache_dir = Path(cache_path).parent
    classifiers = build_classifiers(seed)
    candidate_results = {}
    with tempfile.TemporaryDirectory(prefix="fit-", dir=cache_dir) as temporary_dir:
        train_matrix_path = Path(temporary_dir) / "train_features.npy"
        train_matrix = np.lib.format.open_memmap(
            train_matrix_path, mode="w+", dtype=np.float64, shape=(len(train_indices), 384),
        )
        for start in range(0, len(train_indices), 4096):
            selection = train_indices[start:start + 4096]
            train_matrix[start:start + len(selection)] = features[selection]
        train_matrix.flush()
        y_train = labels[train_indices]
        for number, (name, classifier) in enumerate(classifiers.items()):
            progress({"stage": f"training_{name}", "progress": 0.72 + number * 0.06,
                      "message": f"Fitting {name.replace('_', ' ')} on the training split."})
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                classifier.fit(train_matrix, y_train)
            fit_seconds = time.perf_counter() - started
            validation_predictions, validation_inference_seconds = _predict_batched(
                classifier, features, validation_indices,
            )
            test_predictions, inference_seconds = _predict_batched(classifier, features, test_indices)
            test_scores, score_labels = _scores_batched(classifier, features, test_indices)
            candidate_results[name] = {
                "model": classifier,
                "training_time_seconds": round(fit_seconds, 4),
                "validation_inference_time_seconds": round(validation_inference_seconds, 4),
                "test_inference_time_seconds": round(inference_seconds, 4),
                "validation": _metric_report(labels[validation_indices], validation_predictions),
                "test": _metric_report(labels[test_indices], test_predictions, test_scores, score_labels),
                "warnings": [str(item.message) for item in caught],
                "configuration": classifier.get_params(),
            }
        # Windows cannot delete the temporary directory while the memmap is open.
        del train_matrix
        gc.collect()

    # Deep Learning supervised classifier on the same frozen SBERT features and splits.
    train_x = np.asarray(features[train_indices], dtype=np.float32)
    validation_x = np.asarray(features[validation_indices], dtype=np.float32)
    test_x = np.asarray(features[test_indices], dtype=np.float32)
    feature_mean, feature_std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    progress({"stage": "training_mlp", "progress": 0.9, "message": "Fitting PyTorch MLP on the training split."})
    started = time.perf_counter()
    mlp_model, mlp_training = train_mlp(train_x, labels[train_indices], validation_x, labels[validation_indices],
                                        feature_mean, feature_std, seed=seed, progress=progress)
    mlp_fit_seconds = time.perf_counter() - started
    mlp_classifier = MLPPredictor(mlp_model, feature_mean, feature_std)
    started = time.perf_counter()
    mlp_validation_predictions = mlp_classifier.predict(validation_x)
    mlp_validation_inference = time.perf_counter() - started
    started = time.perf_counter()
    mlp_test_predictions = mlp_classifier.predict(test_x)
    mlp_inference = time.perf_counter() - started
    candidate_results["mlp"] = {
        "model": mlp_classifier,
        "model_category": MODEL_CATEGORIES["mlp"],
        "training_time_seconds": round(mlp_fit_seconds, 4),
        "validation_inference_time_seconds": round(mlp_validation_inference, 4),
        "test_inference_time_seconds": round(mlp_inference, 4),
        "model_size_bytes": model_size_bytes(mlp_model),
        "validation": _metric_report(labels[validation_indices], mlp_validation_predictions),
        "test": _metric_report(labels[test_indices], mlp_test_predictions, mlp_classifier.predict_proba(test_x),
                               MLP_LABELS),
        "warnings": [],
        "configuration": mlp_training,
    }
    best_name = max(
        candidate_results,
        key=lambda name: (candidate_results[name]["validation"]["macro_f1"],
                          candidate_results[name]["validation"]["weighted_f1"]),
    )
    selected = candidate_results[best_name]
    sklearn_name = max(
        ("logistic_regression", "linear_svm"),
        key=lambda name: (candidate_results[name]["validation"]["macro_f1"],
                          candidate_results[name]["validation"]["weighted_f1"]),
    )
    for name in candidate_results:
        if name != "sbert_reference_baseline" and "model_size_bytes" not in candidate_results[name]:
            candidate_results[name]["model_size_bytes"] = _sklearn_size(candidate_results[name]["model"])
    metadata = {
        "selected_model": best_name,
        "default_classifier": best_name,
        "best_traditional_classifier": sklearn_name,
        "classifier_files": {"mlp": "mlp/mlp_classifier.pt", sklearn_name: "classifier.joblib"},
        "selected_model_path": str(training_dir / "classifier.joblib"),
        "selected_by": "highest validation macro F1 among supervised classifiers; weighted F1 breaks ties",
        "trained_classifier": selected["configuration"],
        "sbert_model_id": MODEL_ID,
        "sbert_package_version": model_version,
        "label_mapping": LABEL_MAPPING,
        "preprocessing": PREPROCESSING,
        "training_dataset_fingerprint_sha256": audit["dataset_fingerprint_sha256"],
        "split": split_info,
        "evaluation_label_source": "rating-derived weak labels; not verified sentiment ground truth",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_cache_path": str(cache_path),
        "embedding_cache_reused": reused_cache,
        "model_type": "trained linear classifier over frozen SBERT embeddings",
    }
    results = {
        "status": "completed",
        "dataset_audit": audit,
        "split": split_info,
        "model_comparison": {
            "sbert_reference_baseline": {
                "model_category": MODEL_CATEGORIES["sbert_reference_baseline"],
                "model_size_bytes": int((root / "models" / "reference_embeddings.npy").stat().st_size)
                if (root / "models" / "reference_embeddings.npy").is_file() else 0,
                "training_time_seconds": 0.0,
                "test_inference_time_seconds": round(baseline_inference_seconds, 4),
                "test": baseline_report,
                "configuration": {"method": "maximum cosine similarity to fixed reference phrase embeddings"},
            },
            **{
                name: {"model_category": MODEL_CATEGORIES[name],
                       **{key: value for key, value in result.items() if key not in ("model", "model_category")}}
                for name, result in candidate_results.items()
            },
        },
        "selected_model": best_name,
        "selected_model_test": selected["test"],
        "metadata": metadata,
    }
    for name in candidate_results:
        candidate_results[name].setdefault("model_category", MODEL_CATEGORIES[name])
    save_selected_classifier(candidate_results[sklearn_name]["model"], training_dir, metadata)
    save_mlp(training_dir / "mlp", mlp_classifier.model, feature_mean, feature_std, {
        "model_category": MODEL_CATEGORIES["mlp"], "training": mlp_training,
        "label_mapping": LABEL_MAPPING, "feature_standardisation": "training-split mean and std",
        "test_metrics": candidate_results["mlp"]["test"],
    })
    _write_json(training_dir / "training_report.json", results)
    progress({"stage": "completed", "progress": 1.0, "message": f"Saved selected model: {best_name}."})
    del features
    gc.collect()
    return results
