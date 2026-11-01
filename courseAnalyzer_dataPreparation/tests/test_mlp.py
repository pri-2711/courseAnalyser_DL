import numpy as np
import torch

from src import mlp


def test_architecture_matches_specification_and_returns_logits():
    model = mlp.SentimentMLP()
    layers = list(model.network)
    linear_shapes = [(layer.in_features, layer.out_features) for layer in layers if isinstance(layer, torch.nn.Linear)]
    assert linear_shapes == [(384, 256), (256, 128), (128, 64), (64, 3)]
    batchnorms = [layer.num_features for layer in layers if isinstance(layer, torch.nn.BatchNorm1d)]
    assert batchnorms == [256, 128]
    dropouts = [layer.p for layer in layers if isinstance(layer, torch.nn.Dropout)]
    assert dropouts == [0.3, 0.2, 0.1]
    model.eval()
    logits = model(torch.zeros(4, 384))
    assert logits.shape == (4, 3)
    probabilities = torch.softmax(logits, dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4))


def test_single_sample_is_safe_in_eval_mode_and_skipped_in_training_batches():
    model = mlp.SentimentMLP().eval()
    assert model(torch.zeros(1, 384)).shape == (1, 3)
    assert list(mlp._batches(5, 2)) == [[0, 1], [2, 3]]


def test_class_weights_come_from_training_labels_only():
    labels = np.array(["negative"] * 2 + ["neutral"] * 2 + ["positive"] * 8)
    weights, mapping = mlp.balanced_class_weights(labels)
    assert weights[0] > weights[2]
    assert mapping["positive"] < mapping["negative"]


def test_train_save_and_reload_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    labels = np.array(["negative", "neutral", "positive"])
    train_y = labels[np.arange(120) % 3]
    val_y = labels[np.arange(60) % 3]
    train_x = rng.normal(size=(120, 384)).astype(np.float32) + (np.arange(120)[:, None] % 3)
    val_x = rng.normal(size=(60, 384)).astype(np.float32) + (np.arange(60)[:, None] % 3)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    model, info = mlp.train_mlp(train_x, train_y, val_x, val_y, mean, std, max_epochs=3, patience=5, batch_size=32)
    assert info["epochs_run"] == 3
    assert info["optimizer"]["learning_rate"] == 0.001
    assert info["scheduler"]["monitor"] == "validation loss"
    assert len(info["history"]["train_loss"]) == 3
    mlp.save_mlp(tmp_path, model, mean, std, {"training": info})
    reloaded = mlp.load_mlp(tmp_path)
    original = mlp.MLPPredictor(model, mean, std).predict_proba(val_x)
    assert np.allclose(original, reloaded.predict_proba(val_x), atol=1e-6)
    assert reloaded.predict(val_x[:5]).shape == (5,)
