"""Numerical and integration tests for streaming ridge probe."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from open_eeg_bench.default_configs.datasets import ALL_DATASETS


@pytest.fixture
def classif_data():
    """Synthetic linearly separable data for classification."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N, D, C = 500, 20, 3
    W_true = rng.standard_normal((D, C))
    X = rng.standard_normal((N, D)).astype(np.float32)
    logits = X @ W_true
    y = logits.argmax(axis=1).astype(np.int64)
    # Split 60/20/20
    i1, i2 = int(0.6 * N), int(0.8 * N)
    return {
        "train": (X[:i1], y[:i1]),
        "val":   (X[i1:i2], y[i1:i2]),
        "test":  (X[i2:], y[i2:]),
        "D": D, "C": C,
    }


def _loader(X, y, batch_size=32):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_fit_streaming_ridge_matches_sklearn_classification(classif_data):
    """Streaming ridge weights match sklearn.Ridge on one-hot targets."""
    from sklearn.linear_model import Ridge
    from open_eeg_bench.ridge_probe import _fit_streaming_ridge

    X_tr, y_tr = classif_data["train"]
    X_val, y_val = classif_data["val"]
    C = classif_data["C"]
    lam = 1.0

    out = _fit_streaming_ridge(
        model=nn.Identity(),
        train_loader=_loader(X_tr, y_tr),
        val_loader=_loader(X_val, y_val),
        n_classes=C,
        lambdas=[lam],
        device="cpu",
    )

    # sklearn reference: one-hot Ridge with fit_intercept=True
    Y_oh = np.eye(C)[y_tr]
    ref = Ridge(alpha=lam, fit_intercept=True).fit(X_tr, Y_oh)

    W = out["W"].cpu().numpy()       # (D, C)
    b = out["bias"].cpu().numpy()    # (C,)
    np.testing.assert_allclose(W, ref.coef_.T, atol=1e-4)
    np.testing.assert_allclose(b, ref.intercept_, atol=1e-4)


def test_fit_streaming_ridge_selects_best_lambda(classif_data):
    """λ sweep picks a valid index with populated val scores."""
    from open_eeg_bench.ridge_probe import _fit_streaming_ridge

    X_tr, y_tr = classif_data["train"]
    X_val, y_val = classif_data["val"]
    C = classif_data["C"]
    lambdas = [1e-2, 1.0, 1e2]

    out = _fit_streaming_ridge(
        model=nn.Identity(),
        train_loader=_loader(X_tr, y_tr),
        val_loader=_loader(X_val, y_val),
        n_classes=C,
        lambdas=lambdas,
        device="cpu",
    )
    assert out["best_lambda"] in lambdas
    assert set(out["val_scores"].keys()) == set(lambdas)
    # Should classify better than chance on separable data
    best_score = out["val_scores"][out["best_lambda"]]
    assert best_score > 1.0 / C


@pytest.fixture
def regression_data():
    """Synthetic linear regression data."""
    rng = np.random.default_rng(1)
    N, D = 500, 15
    w_true = rng.standard_normal((D, 1))
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = (X @ w_true + 0.1 * rng.standard_normal((N, 1))).astype(np.float32).reshape(-1)
    i1, i2 = int(0.6 * N), int(0.8 * N)
    return {
        "train": (X[:i1], y[:i1]),
        "val":   (X[i1:i2], y[i1:i2]),
        "test":  (X[i2:], y[i2:]),
        "D": D,
    }


def test_fit_streaming_ridge_regression_matches_sklearn(regression_data):
    """Streaming ridge weights match sklearn.Ridge on regression targets."""
    from sklearn.linear_model import Ridge
    from open_eeg_bench.ridge_probe import _fit_streaming_ridge

    X_tr, y_tr = regression_data["train"]
    X_val, y_val = regression_data["val"]
    lam = 0.5

    out = _fit_streaming_ridge(
        model=nn.Identity(),
        train_loader=_loader(X_tr, y_tr),
        val_loader=_loader(X_val, y_val),
        n_classes=None,   # regression
        lambdas=[lam],
        device="cpu",
    )

    ref = Ridge(alpha=lam, fit_intercept=True).fit(X_tr, y_tr.reshape(-1, 1))
    W = out["W"].cpu().numpy()
    b = out["bias"].cpu().numpy()
    np.testing.assert_allclose(W.ravel(), ref.coef_.ravel(), atol=1e-4)
    np.testing.assert_allclose(b.ravel(), ref.intercept_.ravel(), atol=1e-4)
    # Val R² should be high on noise-free-ish synthetic data
    assert out["val_scores"][lam] > 0.9


def test_learner_fit_predict_classification(classif_data):
    from open_eeg_bench.ridge_probe import StreamingRidgeProbeLearner

    X_tr, y_tr = classif_data["train"]
    X_val, y_val = classif_data["val"]
    X_te, y_te = classif_data["test"]
    C = classif_data["C"]

    val_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_set = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_set  = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))

    learner = StreamingRidgeProbeLearner(
        feature_extractor=nn.Identity(),
        n_classes=C,
        batch_size=32,
        num_workers=0,
        device="cpu",
        lambdas=[1e-2, 1.0, 1e2],
        val_set=val_set,
    )
    learner.fit(train_set, y=None)
    preds = learner.predict(test_set)
    assert preds.shape == (len(X_te),)
    assert preds.dtype in (np.int64, np.int32)
    # Accuracy above chance
    acc = (preds == y_te).mean()
    assert acc > 1.0 / C


def test_learner_fit_predict_regression(regression_data):
    from open_eeg_bench.ridge_probe import StreamingRidgeProbeLearner

    X_tr, y_tr = regression_data["train"]
    X_val, y_val = regression_data["val"]
    X_te, y_te = regression_data["test"]

    val_set   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_set = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_set  = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))

    learner = StreamingRidgeProbeLearner(
        feature_extractor=nn.Identity(),
        n_classes=None,
        batch_size=32,
        num_workers=0,
        device="cpu",
        lambdas=None,
        val_set=val_set,
    )
    learner.fit(train_set, y=None)
    preds = learner.predict(test_set)
    assert preds.shape == (len(X_te), 1) or preds.shape == (len(X_te),)


@pytest.mark.slow
@pytest.mark.parametrize("dataset_name", list(ALL_DATASETS.keys()))
def test_ridge_probe_end_to_end(dataset_name):
    """Full Experiment.run() with ridge_probe on BIOT for each dataset.

    Skips datasets not already downloaded locally.
    """
    from pathlib import Path
    from open_eeg_bench.default_configs.backbones import biot
    from open_eeg_bench.experiment import Experiment
    from open_eeg_bench.finetuning import Frozen
    from open_eeg_bench.head import FlattenHead
    from open_eeg_bench.training import RidgeProbingTraining

    dataset_cfg = ALL_DATASETS[dataset_name]()

    cache_dir = f"datasets--{dataset_cfg.hf_id.replace('/', '--')}"
    cache_path = Path.home() / ".cache" / "huggingface" / "hub" / cache_dir
    if not cache_path.exists():
        pytest.skip(f"{dataset_cfg.hf_id} not downloaded locally")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp = Experiment(
        backbone=biot(),
        head=FlattenHead(),
        finetuning=Frozen(),
        dataset=dataset_cfg,
        training=RidgeProbingTraining(device=device, batch_size=32),
        seed=0,
    )
    result = exp.run()

    if dataset_cfg.n_classes is None:
        assert "test_r2" in result
    else:
        chance = 1.0 / dataset_cfg.n_classes
        assert "test_balanced_accuracy" in result
        assert result["test_balanced_accuracy"] > chance
