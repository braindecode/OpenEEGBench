"""Streaming ridge regression probe for frozen backbone evaluation.

Accumulates sufficient statistics (X^TX, X^TY, centering sums) in float64
over a single pass, eigendecomposes the centered covariance once, then
sweeps λ cheaply in the eigenbasis for each val/test pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.utils.data import DataLoader


def _encode_targets(y: torch.Tensor, n_classes: int | None) -> torch.Tensor:
    """One-hot encode class labels, or reshape regression targets to 2D."""
    if n_classes is None:
        # Regression: ensure (B, C) shape
        return y.float().reshape(y.shape[0], -1)
    # Classification: one-hot
    return torch.nn.functional.one_hot(y.long(), num_classes=n_classes).float()


def _default_lambdas() -> list[float]:
    """Default λ grid: 17 fixed values. Features are standardized internally,
    so eigenvalues of the correlation matrix are O(1) and a fixed grid works."""
    return [10**e for e in range(-8, 9)]


_STR_TO_DTYPE = {"float32": torch.float32, "float64": torch.float64}


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype not in _STR_TO_DTYPE:
        raise ValueError(
            f"dtype must be 'float32' or 'float64', got {dtype!r}."
        )
    return _STR_TO_DTYPE[dtype]


def _balanced_class_weights(
    train_loader: "DataLoader", n_classes: int, device: str, dtype: torch.dtype
) -> torch.Tensor:
    """One label-only pass over train; returns sklearn-style "balanced" weights.

    ``w[c] = N / (n_classes * count[c])`` so that ``sum_i w_{y_i} == N`` and the
    effective sample size is preserved. Empty classes are clamped to 1 sample
    to avoid div-by-zero (their weight then has no effect on accumulation).
    """
    counts = torch.zeros(n_classes, dtype=dtype, device=device)
    for batch in train_loader:
        y = batch[1].to(device).long()
        counts += torch.bincount(y, minlength=n_classes).to(counts.dtype)
    return counts.sum() / (n_classes * counts.clamp(min=1.0))


def _make_projection_matrix(
    n_features: int, n_components: int, seed: int, device: str, dtype: torch.dtype
) -> torch.Tensor:
    """Return a (n_components, n_features) Gaussian random projection matrix.

    Uses sklearn's ``GaussianRandomProjection._make_random_matrix`` directly so
    that scaling matches scikit-learn exactly (entries ~ N(0, 1/n_components)).
    The private method is called on purpose: ``fit`` requires a numpy array of
    real data, which we don't have at this stage (only the feature dimension).
    """
    from sklearn.random_projection import GaussianRandomProjection

    rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
    P = rp._make_random_matrix(n_components, n_features)
    return torch.from_numpy(np.asarray(P)).to(device=device, dtype=dtype)


def _fit_streaming_ridge(
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    n_classes: int | None,
    lambdas: list[float] | None,
    device: str,
    max_features: int | None = None,
    projection_seed: int = 0,
    class_weight: str | None = "balanced",
    dtype: str = "float64",
) -> dict:
    """Fit streaming ridge probe, select λ on val, return weights + diagnostics.

    If ``max_features`` is set and the backbone emits more features than that,
    features are projected down to ``max_features`` dimensions via a Gaussian
    random projection (seeded by ``projection_seed``) before accumulation.

    ``class_weight`` controls per-sample weighting in the weighted-least-squares
    fit. Only meaningful when ``n_classes`` is set (classification); silently
    ignored for regression. Supported values:

    * ``"balanced"`` (default): sklearn-style ``N / (n_classes * count[c])``
      weights. Costs one extra label-only pass over the train loader.
    * ``None``: unweighted (every sample contributes equally).

    ``dtype`` controls the precision of all internal accumulators, the
    eigendecomposition, and the returned weights. ``"float64"`` (default) is
    recommended for numerical precision: covariances and eigendecompositions
    can lose meaningful accuracy in single precision. Use ``"float32"`` only
    when necessary, e.g. on devices like Apple's MPS that do not support
    float64.

    Returns dict with keys: W (D,C), bias (C,), best_lambda (float),
    val_scores (dict λ→score), lambdas (list[float]), n_classes, n_features D
    (after projection if applied), projection (torch.Tensor | None).
    """
    model.eval()
    model.to(device)
    torch_dtype = _resolve_dtype(dtype)

    # ----- Pass 0 (optional, classification only): per-class weights from labels -----
    class_weights = None
    if class_weight == "balanced" and n_classes is not None:
        class_weights = _balanced_class_weights(train_loader, n_classes, device, torch_dtype)
    elif class_weight not in (None, "balanced"):
        raise ValueError(
            f"class_weight must be 'balanced' or None, got {class_weight!r}."
        )

    # ----- Pass 1: accumulate (optionally weighted) sufficient statistics on train -----
    A = B = s_h = s_h2 = s_y = None
    projection = None  # (k, D_orig); lazily built once D_orig is known
    N = 0.0  # sum of sample weights (== n_samples when unweighted)
    with torch.no_grad():
        for batch in train_loader:
            x, y = batch[0], batch[1]
            h = model(x.to(device))
            h_acc = h.to(dtype=torch_dtype)
            if (
                projection is None
                and max_features is not None
                and h_acc.shape[1] > max_features
            ):
                projection = _make_projection_matrix(
                    n_features=h_acc.shape[1],
                    n_components=max_features,
                    seed=projection_seed,
                    device=device,
                    dtype=torch_dtype,
                )
            if projection is not None:
                h_acc = h_acc @ projection.T  # (B, k)
            y_dev = y.to(device)
            y_enc = _encode_targets(y_dev, n_classes)
            y_acc = y_enc.to(dtype=torch_dtype)

            if class_weights is not None:
                w = class_weights[y_dev.long()]  # (B,)
            else:
                w = torch.ones(h_acc.shape[0], dtype=torch_dtype, device=device)
            hw = h_acc * w.unsqueeze(1)  # (B, D), each row h_i scaled by w_i
            yw = y_acc * w.unsqueeze(1)  # (B, C)

            if A is None:
                D = h_acc.shape[1]
                C = y_acc.shape[1]
                A = torch.zeros(D, D, dtype=torch_dtype, device=device)
                B = torch.zeros(D, C, dtype=torch_dtype, device=device)
                s_h = torch.zeros(D, dtype=torch_dtype, device=device)
                s_h2 = torch.zeros(D, dtype=torch_dtype, device=device)
                s_y = torch.zeros(C, dtype=torch_dtype, device=device)

            A += hw.T @ h_acc
            B += hw.T @ y_acc
            s_h += hw.sum(0)
            s_h2 += (hw * h_acc).sum(0)
            s_y += yw.sum(0)
            N += float(w.sum().item())

    if A is None:
        raise ValueError("Empty train_loader — no features accumulated.")

    if not torch.isfinite(A).all():
        raise RuntimeError(
            "Non-finite (NaN/Inf) values in the accumulated feature covariance. "
            "The backbone is emitting NaN/Inf features — check for RuntimeWarnings "
            "like 'divide by zero' or 'invalid value' during the forward pass. "
            "Ridge probing cannot proceed on polluted features."
        )

    D = A.shape[0]
    C = B.shape[1]

    # ----- Center -----
    h_bar = s_h / N
    y_bar = s_y / N
    C_xx = A - N * torch.outer(h_bar, h_bar)
    C_xy = B - N * torch.outer(h_bar, y_bar)

    # ----- Standardize features (z = (h - mean) / std) -----
    # Work on the correlation matrix instead of the covariance so that
    # eigenvalues are O(1) and a fixed λ grid is meaningful.
    h_std = (s_h2 / N - h_bar**2).sqrt().clamp(min=1e-12)  # (D,)
    inv_std = 1.0 / h_std
    C_zz = C_xx * torch.outer(inv_std, inv_std)  # correlation matrix
    C_zy = C_xy * inv_std.unsqueeze(1)

    # ----- Eigendecomposition (once, on correlation matrix) -----
    # Some accelerators (notably Apple MPS) don't implement torch.linalg.eigh.
    # The decomposition runs on a (D, D) matrix at most ``max_features`` wide,
    # which is small and fast on CPU — do it there unconditionally and move the
    # resulting weights back to ``device`` before the val pass.
    C_zz_cpu = C_zz.cpu()
    C_zy_cpu = C_zy.cpu()
    inv_std_cpu = inv_std.cpu()
    h_bar_cpu = h_bar.cpu()
    y_bar_cpu = y_bar.cpu()
    eigvals, Q = torch.linalg.eigh(C_zz_cpu)
    Ct = Q.T @ C_zy_cpu  # (D, C)

    if lambdas is None:
        lambdas = _default_lambdas()

    # ----- Solve for each λ in eigenbasis, convert back to original space -----
    K = len(lambdas)
    Ws_cpu = torch.zeros(K, D, C, dtype=torch_dtype)
    biases_cpu = torch.zeros(K, C, dtype=torch_dtype)
    for k, lam in enumerate(lambdas):
        denom = (eigvals + lam).unsqueeze(1)  # (D, 1)
        W_z = Q @ (Ct / denom)  # (D, C) in standardized space
        Ws_cpu[k] = W_z * inv_std_cpu.unsqueeze(1)  # back to original: w_i / std_i
        biases_cpu[k] = y_bar_cpu - Ws_cpu[k].T @ h_bar_cpu  # (C,)
    Ws = Ws_cpu.to(device)
    biases = biases_cpu.to(device)

    # ----- Pass 2: streaming λ selection on val -----
    val_scores = _streaming_val_scores(
        model,
        val_loader,
        Ws,
        biases,
        n_classes=n_classes,
        y_bar_train=y_bar,
        device=device,
        projection=projection,
        dtype=torch_dtype,
    )  # tensor of shape (K,)

    # Among tied best scores, pick the largest λ (most regularization):
    # when val cannot discriminate (e.g. discretized scores on small val sets,
    # plateaus), the most regularized solution is the safest bet.
    # NaN-aware: a NaN val score means the val pass hit non-finite features
    # or predictions — exclude it from both max and tie-breaking.
    finite_mask = torch.isfinite(val_scores)
    if not finite_mask.any():
        raise RuntimeError(
            "All val scores are non-finite — the backbone produced NaN/Inf "
            "features or the ridge solve diverged. Check for RuntimeWarnings "
            "during the forward pass."
        )
    best_score = val_scores[finite_mask].max()
    tied_mask = (val_scores == best_score) & finite_mask
    # lambdas are sorted ascending, so the last tied index is the largest λ
    best_k = int(tied_mask.nonzero()[-1].item())
    return {
        "W": Ws[best_k],
        "bias": biases[best_k],
        "best_lambda": lambdas[best_k],
        "val_scores": {lam: float(val_scores[k]) for k, lam in enumerate(lambdas)},
        "lambdas": lambdas,
        "n_classes": n_classes,
        "n_features": D,
        "projection": projection,
    }


def _streaming_val_scores(
    model: "nn.Module",
    val_loader: "DataLoader",
    Ws: torch.Tensor,  # (K, D, C)
    biases: torch.Tensor,  # (K, C)
    n_classes: int | None,
    y_bar_train: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate per-λ metric streaming on val. Returns (K,) scores (higher=better).

    `y_bar_train` is only used by the regression branch (for SS_tot reference);
    ignored for classification (balanced accuracy doesn't need it).
    """
    K, _, C = Ws.shape
    is_regression = n_classes is None

    if is_regression:
        # R² = 1 - SS_res / SS_tot  (SS_tot using train mean; matches sklearn behavior closely enough)
        ss_res = torch.zeros(K, dtype=dtype, device=device)
        ss_tot_scalar = torch.zeros((), dtype=dtype, device=device)
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0], batch[1]
                h = model(x.to(device)).to(dtype=dtype)
                if projection is not None:
                    h = h @ projection.T
                y_enc = _encode_targets(y.to(device), n_classes).to(dtype=dtype)
                preds = torch.einsum("kdc,bd->kbc", Ws, h) + biases.unsqueeze(1)
                res = preds - y_enc.unsqueeze(0)
                ss_res += (res**2).sum(dim=(1, 2))
                ss_tot_scalar += ((y_enc - y_bar_train) ** 2).sum()
        return 1.0 - ss_res / ss_tot_scalar.clamp(min=1e-12)

    # Classification: balanced accuracy via confusion matrix (K, C, C)
    confusion = torch.zeros(K, C, C, dtype=dtype, device=device)
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0], batch[1]
            h = model(x.to(device)).to(dtype=dtype)
            if projection is not None:
                h = h @ projection.T
            y_true = y.to(device).long()
            preds = torch.einsum("kdc,bd->kbc", Ws, h) + biases.unsqueeze(1)
            y_pred = preds.argmax(dim=2)  # (K, B)
            linear_idx = y_true.unsqueeze(0) * C + y_pred  # (K, B)
            offsets = (
                torch.arange(K, device=device, dtype=linear_idx.dtype).unsqueeze(1) * (C * C)
            )  # (K, 1)
            counts = torch.bincount(
                (linear_idx + offsets).reshape(-1),
                minlength=K * C * C,
            ).reshape(K, C, C)
            confusion += counts.to(dtype=confusion.dtype)

    # Balanced accuracy = mean of per-class recall
    per_class = confusion.sum(dim=2).clamp(min=1)
    recalls = confusion.diagonal(dim1=1, dim2=2) / per_class
    return recalls.mean(dim=1)


class StreamingRidgeProbeLearner:
    """skorch-compatible adapter: .fit(train_set, y=None), .predict(test_set).

    Used by RidgeProbingTraining.build_learner. Expects feature_extractor to
    produce (B, D) features (FlattenHead is applied upstream in Experiment).
    """

    def __init__(
        self,
        feature_extractor,
        n_classes: int | None,
        batch_size: int,
        num_workers: int,
        device: str,
        lambdas: list[float] | None,
        val_set,
        max_features: int | None = None,
        projection_seed: int = 0,
        class_weight: str | None = "balanced",
        dtype: str = "float64",
        verbose: int = 1,
    ):
        self.model_ = feature_extractor
        self.n_classes_ = n_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.lambdas = lambdas
        self.val_set = val_set
        self.max_features = max_features
        self.projection_seed = projection_seed
        self.class_weight = class_weight
        self.dtype = dtype
        self.verbose = verbose
        self._result: dict | None = None

    def fit(self, train_set, y=None):
        from torch.utils.data import DataLoader

        self.model_.eval()
        self.model_.to(self.device)

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

        self._result = _fit_streaming_ridge(
            model=self.model_,
            train_loader=train_loader,
            val_loader=val_loader,
            n_classes=self.n_classes_,
            lambdas=self.lambdas,
            device=self.device,
            max_features=self.max_features,
            projection_seed=self.projection_seed,
            class_weight=self.class_weight,
            dtype=self.dtype,
        )
        if self.verbose:
            metric = "R²" if self.n_classes_ is None else "balanced_acc"
            lines = [f"Ridge probe fit complete (metric: {metric})"]
            lines.append(f"  {'lambda':>12s}  {metric:>12s}  {'selected':>8s}")
            lines.append(f"  {'─' * 12}  {'─' * 12}  {'─' * 8}")
            best_lam = self._result["best_lambda"]
            for lam, score in self._result["val_scores"].items():
                marker = "  ←" if lam == best_lam else ""
                lines.append(f"  {lam:12.3g}  {score:12.4f}{marker}")
            print("\n".join(lines))
        return self

    def predict(self, test_set) -> np.ndarray:
        from torch.utils.data import DataLoader

        if self._result is None:
            raise RuntimeError("Call .fit() before .predict().")

        W = self._result["W"]  # (D, C)
        bias = self._result["bias"]  # (C,)
        projection = self._result.get("projection")  # (k, D_orig) or None
        torch_dtype = _resolve_dtype(self.dtype)
        loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

        self.model_.eval()
        self.model_.to(self.device)
        outs = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                h = self.model_(x.to(self.device)).to(dtype=torch_dtype)
                if projection is not None:
                    h = h @ projection.T
                y_hat = h @ W + bias  # (B, C)
                outs.append(y_hat.cpu().numpy())
        preds = np.concatenate(outs, axis=0)

        if self.n_classes_ is None:
            return preds  # (N, C_out)
        return preds.argmax(axis=1).astype(np.int64)
