"""One-time backfill of MLP / linear-SVM labels for saved course analyses.

Uses the saved classifiers and the embeddings already stored per course. Nothing is retrained and no
embeddings are regenerated. Only blank prediction cells are filled, and each course summary is
recomputed from the updated predictions.

Run from the project directory:  .venv\\Scripts\\python.exe scripts\\backfill_supervised.py
"""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app import database, main  # noqa: E402
from app.backfill import backfill_supervised_labels  # noqa: E402


def main_entry():
    classifiers = main._supervised()[1]
    with database.connect() as db:
        rows = db.execute("SELECT id, reviews_path FROM analyses").fetchall()
    for row in rows:
        predictions = Path(row["reviews_path"]) if row["reviews_path"] else None
        if not predictions or not predictions.is_file():
            print(f"{row['id']}: no predictions file, skipped")
            continue
        frame = pd.read_csv(predictions)
        frame, report = backfill_supervised_labels(frame, predictions, predictions.parent / "embeddings.npy", classifiers)
        if report["status"] == "filled":
            summary = main._summary(frame)
            with database.connect() as db:
                db.execute("UPDATE analyses SET summary_json=? WHERE id=?", (json.dumps(summary), row["id"]))
        print(f"{row['id']}: {report['status']} {report['filled']}")


if __name__ == "__main__":
    main_entry()
