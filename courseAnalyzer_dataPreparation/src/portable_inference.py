"""Load the packaged local encoder and classifiers without retraining."""
from pathlib import Path
import json

import joblib
import numpy as np

from .mlp import load_mlp
from .preprocessing import clean_text
from .sbert_model import ReferenceSBERT


def package_root(path=None):
    return Path(path or Path(__file__).resolve().parents[1] / "outputs" / "models").resolve()


def load_portable_predictor(path=None):
    """Return (encoder, {classifier_name: classifier}, metadata). Classifiers never train here."""
    root = package_root(path)
    metadata_path = root / "model_metadata.json"
    encoder_path = root / "sbert_model"
    references_path = root / "reference_embeddings.npy"
    if not all(item.exists() for item in (metadata_path, encoder_path, references_path)):
        raise FileNotFoundError("Portable model package is incomplete. Expected outputs/models/{sbert_model,reference_embeddings.npy,model_metadata.json,...}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    encoder = ReferenceSBERT(model_dir=encoder_path, reference_embeddings_path=references_path,
                             require_local=True, batch_size=int(metadata.get("embedding_batch_size", 64)))
    classifiers = {}
    for name, relative in metadata["classifier_files"].items():
        target = root / relative
        if not target.is_file():
            raise FileNotFoundError(f"Packaged classifier missing: {relative}")
        classifiers[name] = load_mlp(target.parent) if name == "mlp" else joblib.load(target)
    return encoder, classifiers, metadata


def predict_texts(texts, path=None, classifier=None):
    return predict_with_predictor(load_portable_predictor(path), texts, classifier)


DISPLAY_DEFAULT_CLASSIFIER = "mlp"


def predict_with_predictor(predictor, texts, classifier=None):
    """Predict with an already-loaded (encoder, classifiers, metadata) tuple; never trains.

    With no name given, the MLP is used (the application default). The training report's
    validation-selected model is unchanged and still reported by /model/status.
    """
    encoder, classifiers, metadata = predictor
    name = classifier or (DISPLAY_DEFAULT_CLASSIFIER if DISPLAY_DEFAULT_CLASSIFIER in classifiers else metadata["default_classifier"])
    if name not in classifiers:
        raise ValueError(f"Unknown classifier '{name}'. Available: {', '.join(sorted(classifiers))}")
    model = classifiers[name]
    cleaned = [clean_text(text) for text in texts]
    embeddings = np.asarray(encoder.embed(cleaned), dtype=np.float32)
    labels = model.predict(embeddings)
    probabilities = model.predict_proba(embeddings) if hasattr(model, "predict_proba") else None
    return [{"review_text": text, "cleaned_text": clean, "classifier": name, "predicted_label": str(label),
             "class_scores": ({str(cls): float(value) for cls, value in zip(model.classes_, row)} if probabilities is not None else None)}
            for text, clean, label, row in zip(texts, cleaned, labels, probabilities if probabilities is not None else [None] * len(texts))]
