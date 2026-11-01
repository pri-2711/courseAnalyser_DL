"""Train and package local weak-label SBERT classifiers with terminal progress."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.training import run_training, load_saved_classifier, LABELS
from src.portable_inference import predict_texts

ROOT = Path(__file__).resolve().parent


class TerminalProgress:
    def __init__(self):
        self.bar = tqdm(total=1000, desc="Training workflow", unit="‰", dynamic_ncols=True)
        self.last = 0

    def __call__(self, event):
        value = min(1000, max(self.last, int(float(event.get("progress", 0)) * 1000)))
        self.bar.update(value - self.last)
        self.last = value
        self.bar.set_postfix_str(event.get("stage", "working"))
        tqdm.write(event.get("message", event.get("stage", "working")))

    def close(self):
        self.bar.update(1000 - self.last)
        self.bar.close()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def create_graphs(report, graphs_dir):
    graphs_dir.mkdir(parents=True, exist_ok=True)
    comparison = report["model_comparison"]
    names = list(comparison)
    labels = [name.replace("_", " ") for name in names]
    values = [comparison[name]["test"]["macro_f1"] for name in names]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, values, color=["#777991", "#5656B5", "#AAA5F5", "#E0708A"][:len(values)])
    axis.set_ylim(0, 1); axis.set_ylabel("Test macro F1"); axis.set_title("Weak-label model comparison")
    axis.bar_label(bars, fmt="%.3f", padding=3); figure.tight_layout()
    figure.savefig(graphs_dir / "model_comparison_macro_f1.png", dpi=180); plt.close(figure)
    for name, result in comparison.items():
        matrix = np.asarray(result["test"]["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(5, 4.2))
        image = axis.imshow(matrix, cmap="Purples")
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        axis.set_xlabel("Predicted weak label"); axis.set_ylabel("Rating-derived weak label")
        axis.set_title(f"{name.replace('_', ' ')} test confusion matrix")
        for row in range(len(LABELS)):
            for col in range(len(LABELS)):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
        figure.colorbar(image, ax=axis); figure.tight_layout()
        figure.savefig(graphs_dir / f"{name}_confusion_matrix.png", dpi=180); plt.close(figure)


def create_mlp_graphs(report, graphs_dir):
    comparison = report["model_comparison"]
    history = comparison["mlp"]["configuration"]["history"]
    epochs = history["epoch"]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, history["train_loss"], marker="o", label="training loss")
    axis.plot(epochs, history["val_loss"], marker="o", label="validation loss")
    axis.set_xlabel("Epoch"); axis.set_ylabel("Cross-entropy loss"); axis.set_title("MLP loss curve")
    axis.legend(); figure.tight_layout(); figure.savefig(graphs_dir / "mlp_loss_curve.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, history["train_accuracy"], marker="o", label="training accuracy")
    axis.plot(epochs, history["val_accuracy"], marker="o", label="validation accuracy")
    axis.plot(epochs, history["val_macro_f1"], marker="s", linestyle="--", label="validation macro F1")
    axis.set_xlabel("Epoch"); axis.set_ylabel("Score"); axis.set_ylim(0, 1); axis.set_title("MLP accuracy and macro F1")
    axis.legend(); figure.tight_layout(); figure.savefig(graphs_dir / "mlp_accuracy_curve.png", dpi=180); plt.close(figure)
    matrix = np.asarray(comparison["mlp"]["test"]["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4.2))
    image = axis.imshow(matrix, cmap="Purples")
    axis.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right"); axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("Predicted weak label"); axis.set_ylabel("Rating-derived weak label"); axis.set_title("MLP test confusion matrix")
    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(graphs_dir / "mlp_confusion_matrix.png", dpi=180); plt.close(figure)
    names = [name for name in comparison if name != "sbert_reference_baseline"]
    width = 0.8 / len(names)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for offset, name in enumerate(names):
        per_class = comparison[name]["test"]["per_class"]
        values = [per_class[label]["f1-score"] for label in LABELS]
        axis.bar(np.arange(len(LABELS)) + offset * width, values, width, label=name.replace("_", " "))
    axis.set_xticks(np.arange(len(LABELS)) + width * (len(names) - 1) / 2, LABELS)
    axis.set_ylim(0, 1); axis.set_ylabel("Test F1"); axis.set_title("Per-class F1 comparison (same test set)")
    axis.legend(); figure.tight_layout(); figure.savefig(graphs_dir / "per_class_f1_comparison.png", dpi=180); plt.close(figure)


def write_reports(report, reports_dir):
    reports_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(reports_dir / "training_report.json", report)
    rows = []
    for name, result in report["model_comparison"].items():
        metrics = result["test"]
        rows.append({"model": name, **{field: metrics.get(field) for field in
                    ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "roc_auc_ovr_macro")},
                    "training_time_seconds": result.get("training_time_seconds", 0.0),
                    "test_inference_time_seconds": result.get("test_inference_time_seconds", 0.0)})
        pd.DataFrame(metrics["per_class"]).T.to_csv(reports_dir / f"{name}_classification_report.csv")
        pd.DataFrame(metrics["confusion_matrix"], index=metrics["confusion_matrix_labels"],
                     columns=metrics["confusion_matrix_labels"]).to_csv(reports_dir / f"{name}_confusion_matrix.csv")
    pd.DataFrame(rows).to_csv(reports_dir / "model_comparison.csv", index=False)
    markdown = ["# Supervised SBERT evaluation", "", "Labels are derived from ratings: 1–2 negative, 3 neutral, 4–5 positive. They are weak labels, not manually verified ground truth.", "",
                "| Model | Accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        markdown.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | {row['roc_auc_ovr_macro'] if row['roc_auc_ovr_macro'] is not None else 'n/a'} |")
    (reports_dir / "evaluation_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def package_models(report, output_root, batch_size):
    models_dir = output_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    training_dir = ROOT / "models" / "trained_sentiment"
    classifier, training_metadata = load_saved_classifier(training_dir)
    classifier_path = models_dir / "sentiment_model.joblib"
    joblib.dump(classifier, classifier_path)
    mlp_source = training_dir / "mlp"
    if not (mlp_source / "mlp_classifier.pt").is_file():
        raise FileNotFoundError(f"MLP checkpoint missing at {mlp_source}; training must complete before packaging.")
    if (models_dir / "mlp").exists():
        shutil.rmtree(models_dir / "mlp")
    shutil.copytree(mlp_source, models_dir / "mlp")
    source_encoder = ROOT / "models" / "all-MiniLM-L6-v2"
    if not source_encoder.is_dir():
        raise FileNotFoundError(f"Local SBERT model missing at {source_encoder}; inference must complete before packaging.")
    target_encoder = models_dir / "sbert_model"
    if target_encoder.exists():
        shutil.rmtree(target_encoder)
    shutil.copytree(source_encoder, target_encoder)
    reference_source = ROOT / "models" / "reference_embeddings.npy"
    if not reference_source.is_file():
        raise FileNotFoundError(f"Local reference embeddings missing at {reference_source}")
    shutil.copy2(reference_source, models_dir / "reference_embeddings.npy")
    sklearn_name = report["metadata"]["best_traditional_classifier"]
    metadata = {"package_format": 2, "model_id": "all-MiniLM-L6-v2", "default_classifier": report["selected_model"],
                "classifier_files": {sklearn_name: "sentiment_model.joblib", "mlp": "mlp/mlp_classifier.pt"},
                "classifier_categories": {sklearn_name: "Traditional Machine Learning", "mlp": "Deep Learning (supervised PyTorch MLP)"},
                "paths": {"classifier": "sentiment_model.joblib", "mlp": "mlp", "encoder": "sbert_model", "reference_embeddings": "reference_embeddings.npy"},
                "label_mapping": report["metadata"]["label_mapping"], "preprocessing": report["metadata"]["preprocessing"],
                "embedding_batch_size": batch_size, "training_label_source": "rating-derived weak labels; not verified sentiment ground truth",
                "relative_paths_only": True}
    atomic_json(models_dir / "model_metadata.json", metadata)
    return models_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/input/reviews.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = (ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_root = ROOT / "outputs" / "training"
    reports_dir, graphs_dir = output_root / "reports", output_root / "graphs"
    print(f"[1/6] Auditing and training from {source}", flush=True)
    progress = TerminalProgress()
    started = time.perf_counter()
    try:
        report = run_training(source, ROOT, progress=progress, seed=args.seed, embedding_batch_size=args.batch_size)
    finally:
        progress.close()
    print("[2/6] Writing metrics and reports", flush=True)
    write_reports(report, reports_dir)
    print("[3/6] Generating PNG evaluation charts", flush=True)
    create_graphs(report, graphs_dir)
    create_mlp_graphs(report, graphs_dir)
    print("[4/6] Packaging trained classifier and local SentenceTransformer", flush=True)
    models_dir = package_models(report, ROOT / "outputs", args.batch_size)
    print("[5/6] Recording environment", flush=True)
    packages = ["numpy", "pandas", "scikit-learn", "joblib", "torch", "sentence-transformers", "transformers", "matplotlib"]
    atomic_json(ROOT / "outputs" / "training" / "reports" / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    })
    print("[6/6] Reloading package and predicting a new sample without training", flush=True)
    sample = predict_texts(["The flavour is excellent and I would buy it again.", "This product was disappointing and poor quality."], models_dir)
    atomic_json(ROOT / "outputs" / "training" / "reports" / "portable_reload_prediction.json", sample)
    print(json.dumps({"elapsed_seconds": round(time.perf_counter() - started, 2), "selected_model": report["selected_model"],
                      "outputs": {"reports": str(reports_dir), "graphs": str(graphs_dir), "models": str(models_dir)}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
