"""Confusion-matrix evaluation against user-supplied manual labels."""
import numpy as np
from .data_loader import _field

LABELS = ("positive", "negative", "neutral")
LABEL_MAP = {"pos": "positive", "2": "positive", "5": "positive",
             "neg": "negative", "-1": "negative", "0": "negative",
             "neu": "neutral", "1": "neutral", "3": "neutral"}


def evaluation_columns(frame):
    text = _field(frame.columns, ("review_clean", "review", "review_text", "text", "comment", "content"))
    label = _field(frame.columns, ("label", "sentiment"))
    if text is None or label is None:
        raise ValueError("Evaluation data needs a review text column and a label/sentiment column.")
    return text, label


class EvaluationAccumulator:
    def __init__(self):
        self.matrix = np.zeros((3, 3), dtype=np.int64)

    def update(self, truths, predictions):
        indices = {label: i for i, label in enumerate(LABELS)}
        for truth, pred in zip(truths, predictions):
            self.matrix[indices[truth], indices[pred]] += 1

    def report(self):
        total = int(self.matrix.sum())
        if not total:
            return None
        per_class = {}
        for i, label in enumerate(LABELS):
            tp = int(self.matrix[i, i])
            precision = tp / int(self.matrix[:, i].sum()) if self.matrix[:, i].sum() else 0.0
            recall = tp / int(self.matrix[i, :].sum()) if self.matrix[i, :].sum() else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                                "support": int(self.matrix[i, :].sum())}
        return {"sample_count": total,
                "accuracy": float(np.trace(self.matrix) / total),
                "macro_precision": float(np.mean([x["precision"] for x in per_class.values()])),
                "macro_recall": float(np.mean([x["recall"] for x in per_class.values()])),
                "macro_f1": float(np.mean([x["f1"] for x in per_class.values()])),
                "per_class": per_class, "confusion_matrix_labels": list(LABELS),
                "confusion_matrix": self.matrix.tolist()}


def normalize_label(value):
    value = str(value).strip().lower()
    return LABEL_MAP.get(value, value if value in LABELS else None)
