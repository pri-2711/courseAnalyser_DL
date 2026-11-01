"""Fill blank MLP / linear-SVM predictions on historical review records.

Older review records were classified before the supervised classifiers were stored. Their
embeddings are already on disk, so the saved classifiers can label them without re-embedding
anything and without any retraining. Existing labels are never overwritten; only empty cells are
filled. If the embedding rows do not line up with the predictions, nothing is written.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd


def backfill_supervised_labels(frame, predictions_path, embeddings_path, classifiers):
    """Return (frame, report). Writes the predictions CSV only when at least one cell was filled."""
    report = {"status": "unchanged", "filled": {}}
    if not classifiers:
        report["status"] = "classifiers_unavailable"
        return frame, report
    if "embedding_row" not in frame or not Path(embeddings_path).is_file():
        report["status"] = "missing_embedding_rows"
        return frame, report
    vectors = np.load(embeddings_path, mmap_mode="r")
    rows = pd.to_numeric(frame["embedding_row"], errors="coerce")
    if rows.isna().any() or len(frame) == 0 or int(rows.max()) >= len(vectors) or int(rows.min()) < 0:
        report["status"] = "embedding_mismatch"
        return frame, report
    frame = frame.copy()
    positions = rows.astype(int).to_numpy()
    for name, classifier in classifiers.items():
        column = f"{name}_label"
        if column not in frame:
            frame[column] = np.nan
        empty = frame[column].isna().to_numpy()
        if not empty.any():
            continue
        index = np.flatnonzero(empty)
        predicted = classifier.predict(np.asarray(vectors[positions[index]], dtype=np.float32))
        frame.iloc[index, frame.columns.get_loc(column)] = predicted.astype(object)
        report["filled"][column] = int(len(index))
    if report["filled"]:
        temporary = Path(predictions_path).with_name(Path(predictions_path).name + ".tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, predictions_path)
        report["status"] = "filled"
    return frame, report
