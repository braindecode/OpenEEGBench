"""Numerical and integration tests for streaming ridge probe."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
