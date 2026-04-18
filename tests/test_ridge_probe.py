"""Numerical and integration tests for streaming ridge probe."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from open_eeg_bench.default_configs.datasets import ALL_DATASETS


@pytest.fixture
def classif_data():
    """Synthetic linearly separable data for classification.

    D is chosen > 100 so that the ``max_features=100`` parametrization
    actually triggers a random projection in tests that exercise it.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N, D, C = 500, 200, 3
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


@pytest.mark.parametrize("max_features", [None, 100])
def test_fit_streaming_ridge_matches_sklearn_classification(classif_data, max_features):
    """Streaming ridge predictions match a sklearn pipeline on the same X.

    The reference is ``[GaussianRandomProjection?] → StandardScaler → Ridge``:
    when ``max_features`` is set, the sklearn pipeline projects using the same
    ``random_state`` as our ``projection_seed``. Identical seeds must yield
    identical projection matrices — so fitting on raw X_tr on both sides and
    predicting on raw X_tr must match end-to-end.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.random_projection import GaussianRandomProjection
    from open_eeg_bench.ridge_probe import _fit_streaming_ridge

    X_tr, y_tr = classif_data["train"]
    X_val, y_val = classif_data["val"]
    C = classif_data["C"]
    lam = 1.0
    projection_seed = 0

    out = _fit_streaming_ridge(
        model=nn.Identity(),
        train_loader=_loader(X_tr, y_tr),
        val_loader=_loader(X_val, y_val),
        n_classes=C,
        lambdas=[lam],
        device="cpu",
        max_features=max_features,
        projection_seed=projection_seed,
    )

    if max_features is not None:
        # make sure the test is actually testing the projection logic
        assert max_features < classif_data["D"]

    # sklearn reference pipeline: same projection (if any) + standardize + ridge.
    steps = []
    if max_features is not None:
        steps.append(
            GaussianRandomProjection(
                n_components=max_features, random_state=projection_seed
            )
        )
    steps.extend([StandardScaler(), Ridge(alpha=lam, fit_intercept=True)])
    ref = make_pipeline(*steps)
    Y_oh = np.eye(C)[y_tr]
    ref.fit(X_tr, Y_oh)

    # Apply our stored projection (if any) then our ridge weights.
    W = out["W"].cpu().numpy()
    b = out["bias"].cpu().numpy()
    P = out["projection"]
    X_proj = X_tr if P is None else X_tr @ P.cpu().numpy().T
    pred_ours = X_proj @ W + b
    pred_ref = ref.predict(X_tr)
    np.testing.assert_allclose(pred_ours, pred_ref, atol=1e-4)


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
    """Streaming ridge predictions match sklearn StandardScaler + Ridge."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
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

    ref = make_pipeline(StandardScaler(), Ridge(alpha=lam, fit_intercept=True))
    ref.fit(X_tr, y_tr.reshape(-1, 1))

    W = out["W"].cpu().numpy()
    b = out["bias"].cpu().numpy()
    pred_ours = (X_tr @ W + b).ravel()
    pred_ref = ref.predict(X_tr).ravel()
    np.testing.assert_allclose(pred_ours, pred_ref, atol=1e-4)
    # Val R² should be high on noise-free-ish synthetic data
    assert out["val_scores"][lam] > 0.9


def test_fit_streaming_ridge_raises_on_nan_features(classif_data):
    """NaN features (e.g. from a buggy backbone) yield a clear error, not IndexError."""
    from open_eeg_bench.ridge_probe import _fit_streaming_ridge

    class NaNModel(nn.Module):
        def forward(self, x):
            # Emit NaN in one feature dim to mimic signal_jepa's divide-by-zero
            out = x.clone()
            out[:, 0] = float("nan")
            return out

    X_tr, y_tr = classif_data["train"]
    X_val, y_val = classif_data["val"]
    C = classif_data["C"]

    with pytest.raises(RuntimeError, match="NaN/Inf"):
        _fit_streaming_ridge(
            model=NaNModel(),
            train_loader=_loader(X_tr, y_tr),
            val_loader=_loader(X_val, y_val),
            n_classes=C,
            lambdas=[1.0],
            device="cpu",
        )


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


def test_max_features_noop_when_D_small(classif_data):
    """max_features >= D → no projection, result identical to unset."""
    from open_eeg_bench.ridge_probe import _fit_streaming_ridge

    X_tr, y_tr = classif_data["train"]
    X_val, y_val = classif_data["val"]
    C, D = classif_data["C"], classif_data["D"]

    out_ref = _fit_streaming_ridge(
        model=nn.Identity(),
        train_loader=_loader(X_tr, y_tr),
        val_loader=_loader(X_val, y_val),
        n_classes=C,
        lambdas=[1.0],
        device="cpu",
    )
    out = _fit_streaming_ridge(
        model=nn.Identity(),
        train_loader=_loader(X_tr, y_tr),
        val_loader=_loader(X_val, y_val),
        n_classes=C,
        lambdas=[1.0],
        device="cpu",
        max_features=D + 10,  # larger than D → projection skipped
    )
    assert out["projection"] is None
    assert out_ref["projection"] is None
    torch.testing.assert_close(out["W"], out_ref["W"])
    torch.testing.assert_close(out["bias"], out_ref["bias"])


def test_max_features_activates_and_reduces_dim(classif_data):
    """max_features < D → projection built, n_features matches target."""
    from open_eeg_bench.ridge_probe import _fit_streaming_ridge

    X_tr, y_tr = classif_data["train"]
    X_val, y_val = classif_data["val"]
    C, D = classif_data["C"], classif_data["D"]
    k = 8  # D=20 → project to 8

    out = _fit_streaming_ridge(
        model=nn.Identity(),
        train_loader=_loader(X_tr, y_tr),
        val_loader=_loader(X_val, y_val),
        n_classes=C,
        lambdas=[1.0],
        device="cpu",
        max_features=k,
    )
    P = out["projection"]
    assert P is not None
    assert P.shape == (k, D)
    assert out["n_features"] == k
    # W is in projected space
    assert out["W"].shape == (k, C)
    # Still classifies above chance on separable synthetic data
    Y_oh = np.eye(C)[y_tr]
    X64 = torch.from_numpy(X_tr).double()
    pred = (X64 @ P.T @ out["W"] + out["bias"]).numpy()
    acc = (pred.argmax(axis=1) == y_tr).mean()
    assert acc > 1.0 / C


def test_max_features_deterministic_across_seeds(classif_data):
    """Same projection_seed ⇒ same weights; different seed ⇒ different weights."""
    from open_eeg_bench.ridge_probe import _fit_streaming_ridge

    X_tr, y_tr = classif_data["train"]
    X_val, y_val = classif_data["val"]
    C = classif_data["C"]
    kwargs = dict(
        model=nn.Identity(),
        train_loader=_loader(X_tr, y_tr),
        val_loader=_loader(X_val, y_val),
        n_classes=C,
        lambdas=[1.0],
        device="cpu",
        max_features=8,
    )
    out_a = _fit_streaming_ridge(**kwargs, projection_seed=0)
    out_b = _fit_streaming_ridge(**kwargs, projection_seed=0)
    out_c = _fit_streaming_ridge(**kwargs, projection_seed=42)

    torch.testing.assert_close(out_a["projection"], out_b["projection"])
    torch.testing.assert_close(out_a["W"], out_b["W"])
    assert not torch.allclose(out_a["projection"], out_c["projection"])
    assert not torch.allclose(out_a["W"], out_c["W"])


def test_learner_max_features_predict(classif_data):
    """Learner applies projection consistently at predict time."""
    from open_eeg_bench.ridge_probe import StreamingRidgeProbeLearner

    X_tr, y_tr = classif_data["train"]
    X_val, y_val = classif_data["val"]
    X_te, y_te = classif_data["test"]
    C = classif_data["C"]

    train_set = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_set = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))

    learner = StreamingRidgeProbeLearner(
        feature_extractor=nn.Identity(),
        n_classes=C,
        batch_size=32,
        num_workers=0,
        device="cpu",
        lambdas=[1e-2, 1.0, 1e2],
        val_set=val_set,
        max_features=10,
        projection_seed=0,
    )
    learner.fit(train_set, y=None)
    assert learner._result["projection"] is not None
    preds = learner.predict(test_set)
    assert preds.shape == (len(X_te),)
    # Above chance on synthetic separable data
    assert (preds == y_te).mean() > 1.0 / C


@pytest.mark.slow
@pytest.mark.parametrize("dataset_name", list(ALL_DATASETS.keys()))
def test_ridge_probe_end_to_end(dataset_name):
    """Full Experiment.run() with ridge_probe on CBraMod for each dataset.

    Skips datasets not already downloaded locally.
    """
    from pathlib import Path
    from open_eeg_bench.default_configs.backbones import cbramod
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
        backbone=cbramod(),
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
