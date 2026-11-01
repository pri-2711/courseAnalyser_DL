"""Supervised PyTorch MLP over frozen 384-dimensional SBERT embeddings (Deep Learning classifier).

SBERT (all-MiniLM-L6-v2) is the Transformer feature extractor; this MLP is the
supervised Deep Learning classifier trained on top of those embeddings.
"""
import copy
import io
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn

MLP_LABELS = ("negative", "neutral", "positive")
SEED = 42


class SentimentMLP(nn.Module):
    """384 -> 256 -> 128 -> 64 -> 3 network; returns raw logits (softmax is applied only at inference)."""

    def __init__(self, input_dim=384, num_classes=len(MLP_LABELS)):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def balanced_class_weights(labels):
    """Inverse-frequency weights computed from training labels only."""
    counts = np.array([np.sum(labels == name) for name in MLP_LABELS], dtype=np.float64)
    weights = counts.sum() / (len(MLP_LABELS) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32), dict(zip(MLP_LABELS, weights.round(4).tolist()))


def _batches(count, batch_size, generator=None):
    order = torch.randperm(count, generator=generator).tolist() if generator is not None else list(range(count))
    for start in range(0, count, batch_size):
        batch = order[start:start + batch_size]
        # BatchNorm1d cannot compute batch statistics from a single sample in training mode.
        if len(batch) > 1:
            yield batch


def _predict_logits(model, features, device, batch_size=8192):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(batch).cpu())
    return torch.cat(outputs).numpy() if outputs else np.zeros((0, len(MLP_LABELS)), dtype=np.float32)


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_mlp(train_x, train_y, val_x, val_y, mean, std, max_epochs=30, patience=5, batch_size=256,
              learning_rate=1e-3, weight_decay=1e-4, seed=SEED, progress=None):
    """Train with Adam, ReduceLROnPlateau on validation loss, and early stopping on validation macro-F1.

    train_x/val_x are raw embeddings; standardisation uses the training-set mean and std only.
    Returns the best-validation-macro-F1 model (restored before returning) and a per-epoch history.
    """
    progress = progress or (lambda _event: None)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device()
    train_norm = ((np.asarray(train_x, dtype=np.float32) - mean) / std).astype(np.float32)
    val_norm = ((np.asarray(val_x, dtype=np.float32) - mean) / std).astype(np.float32)
    label_index = {name: i for i, name in enumerate(MLP_LABELS)}
    y_train = np.asarray([label_index[v] for v in train_y], dtype=np.int64)
    y_val = np.asarray([label_index[v] for v in val_y], dtype=np.int64)
    class_weights, weight_map = balanced_class_weights(np.asarray(train_y))

    model = SentimentMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    generator = torch.Generator().manual_seed(seed)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [],
               "val_macro_f1": [], "learning_rate": [], "epoch_seconds": []}
    best = {"macro_f1": -1.0, "epoch": 0, "state": None}
    epochs_without_gain = 0
    total_started = time.perf_counter()
    x_train_tensor = torch.as_tensor(train_norm)
    y_train_tensor = torch.as_tensor(y_train)
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_total, correct, seen = 0.0, 0, 0
        for batch in _batches(len(y_train), batch_size, generator):
            index = torch.as_tensor(batch)
            inputs = x_train_tensor[index].to(device)
            targets = y_train_tensor[index].to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * len(batch)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += len(batch)
        train_loss = loss_total / max(seen, 1)
        train_accuracy = correct / max(seen, 1)

        val_logits = _predict_logits(model, val_norm, device)
        val_tensor_logits = torch.as_tensor(val_logits)
        val_loss = float(nn.functional.cross_entropy(val_tensor_logits, torch.as_tensor(y_val),
                                                     weight=class_weights).item())
        val_pred = val_logits.argmax(axis=1)
        val_accuracy = float((val_pred == y_val).mean())
        val_macro_f1 = float(f1_score(y_val, val_pred, labels=list(range(len(MLP_LABELS))),
                                      average="macro", zero_division=0))
        learning_rate_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        epoch_seconds = time.perf_counter() - epoch_started

        for key, value in zip(history, (epoch, train_loss, val_loss, train_accuracy, val_accuracy,
                                        val_macro_f1, learning_rate_now, epoch_seconds)):
            history[key].append(float(value) if key != "epoch" else int(value))
        progress({"stage": "mlp", "progress": 0.0,
                  "message": (f"MLP epoch {epoch:02d}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
                              f"val acc {val_accuracy:.4f}, val macro F1 {val_macro_f1:.4f}, lr {learning_rate_now:.6f}, "
                              f"{epoch_seconds:.1f}s")})

        if val_macro_f1 > best["macro_f1"]:
            best = {"macro_f1": val_macro_f1, "epoch": epoch,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                progress({"stage": "mlp", "progress": 0.0,
                          "message": f"Early stopping at epoch {epoch}; best epoch {best['epoch']}."})
                break

    model.load_state_dict(best["state"])
    model.to("cpu").eval()
    return model, {
        "history": history,
        "best_epoch": best["epoch"],
        "best_validation_macro_f1": best["macro_f1"],
        "epochs_run": len(history["epoch"]),
        "total_training_seconds": round(time.perf_counter() - total_started, 4),
        "device": str(device),
        "class_weights": weight_map,
        "optimizer": {"name": "Adam", "learning_rate": learning_rate, "weight_decay": weight_decay},
        "scheduler": {"name": "ReduceLROnPlateau", "monitor": "validation loss", "factor": 0.5,
                      "patience": 2, "min_lr": 1e-5},
        "batch_size": batch_size, "max_epochs": max_epochs, "early_stopping_patience": patience, "seed": seed,
    }


class MLPPredictor:
    """Loaded MLP with the standardisation statistics it was trained with; mirrors the sklearn predict API."""

    classes_ = np.asarray(MLP_LABELS)

    def __init__(self, model, mean, std):
        self.model = model.eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def predict_proba(self, features):
        normalized = ((np.asarray(features, dtype=np.float32) - self.mean) / self.std).astype(np.float32)
        return _softmax(_predict_logits(self.model, normalized, torch.device("cpu")))

    def predict(self, features):
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


def model_size_bytes(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def save_mlp(directory, model, mean, std, training_metadata):
    """Persist the state_dict, architecture config, label mapping, scaler and training configuration."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "architecture": {"input_dim": model.input_dim, "num_classes": model.num_classes,
                         "hidden_sizes": [256, 128, 64], "dropout": [0.3, 0.2, 0.1]},
        "label_to_index": {name: i for i, name in enumerate(MLP_LABELS)},
        "scaler_mean": torch.as_tensor(np.asarray(mean, dtype=np.float32)),
        "scaler_std": torch.as_tensor(np.asarray(std, dtype=np.float32)),
    }
    path = directory / "mlp_classifier.pt"
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    metadata_path = directory / "mlp_metadata.json"
    temporary_json = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary_json.write_text(json.dumps(training_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_json.replace(metadata_path)
    return path


def load_mlp(directory):
    """Load the packaged MLP without any training code path or external download."""
    path = Path(directory) / "mlp_classifier.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = checkpoint["architecture"]
    model = SentimentMLP(architecture["input_dim"], architecture["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return MLPPredictor(model, checkpoint["scaler_mean"].numpy(), checkpoint["scaler_std"].numpy())
