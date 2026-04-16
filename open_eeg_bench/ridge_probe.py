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


def _fit_streaming_ridge(
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    n_classes: int | None,
    lambdas: list[float] | None,
    device: str,
) -> dict:
    """Fit streaming ridge probe, select λ on val, return weights + diagnostics.

    Returns dict with keys: W (D,C), bias (C,), best_lambda (float),
    val_scores (dict λ→score), lambdas (list[float]), n_classes, n_features D.
    """
    model.eval()
    model.to(device)

    # ----- Pass 1: accumulate sufficient statistics on train -----
    A = B = s_h = s_h2 = s_y = None
    N = 0
    with torch.no_grad():
        for batch in train_loader:
            x, y = batch[0], batch[1]
            h = model(x.to(device))
            y_enc = _encode_targets(y.to(device), n_classes)
            h64 = h.double()
            y64 = y_enc.double()

            if A is None:
                D = h64.shape[1]
                C = y64.shape[1]
                A = torch.zeros(D, D, dtype=torch.float64, device=device)
                B = torch.zeros(D, C, dtype=torch.float64, device=device)
                s_h = torch.zeros(D, dtype=torch.float64, device=device)
                s_h2 = torch.zeros(D, dtype=torch.float64, device=device)
                s_y = torch.zeros(C, dtype=torch.float64, device=device)

            A += h64.T @ h64
            B += h64.T @ y64
            s_h += h64.sum(0)
            s_h2 += (h64**2).sum(0)
            s_y += y64.sum(0)
            N += h64.shape[0]

    if A is None:
        raise ValueError("Empty train_loader — no features accumulated.")

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
    eigvals, Q = torch.linalg.eigh(C_zz)
    Ct = Q.T @ C_zy  # (D, C)

    if lambdas is None:
        lambdas = _default_lambdas()

    # ----- Solve for each λ in eigenbasis, convert back to original space -----
    K = len(lambdas)
    Ws = torch.zeros(K, D, C, dtype=torch.float64, device=device)
    biases = torch.zeros(K, C, dtype=torch.float64, device=device)
    for k, lam in enumerate(lambdas):
        denom = (eigvals + lam).unsqueeze(1)  # (D, 1)
        W_z = Q @ (Ct / denom)  # (D, C) in standardized space
        Ws[k] = W_z * inv_std.unsqueeze(1)  # back to original: w_i / std_i
        biases[k] = y_bar - Ws[k].T @ h_bar  # (C,)

    # ----- Pass 2: streaming λ selection on val -----
    val_scores = _streaming_val_scores(
        model,
        val_loader,
        Ws,
        biases,
        n_classes=n_classes,
        y_bar_train=y_bar,
        device=device,
    )  # tensor of shape (K,)

    best_k = int(val_scores.argmax().item())
    return {
        "W": Ws[best_k],
        "bias": biases[best_k],
        "best_lambda": lambdas[best_k],
        "val_scores": {lam: float(val_scores[k]) for k, lam in enumerate(lambdas)},
        "lambdas": lambdas,
        "n_classes": n_classes,
        "n_features": D,
    }


def _streaming_val_scores(
    model: "nn.Module",
    val_loader: "DataLoader",
    Ws: torch.Tensor,  # (K, D, C)
    biases: torch.Tensor,  # (K, C)
    n_classes: int | None,
    y_bar_train: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Accumulate per-λ metric streaming on val. Returns (K,) scores (higher=better).

    `y_bar_train` is only used by the regression branch (for SS_tot reference);
    ignored for classification (balanced accuracy doesn't need it).
    """
    K, _, C = Ws.shape
    is_regression = n_classes is None

    if is_regression:
        # R² = 1 - SS_res / SS_tot  (SS_tot using train mean; matches sklearn behavior closely enough)
        ss_res = torch.zeros(K, dtype=torch.float64, device=device)
        ss_tot_scalar = torch.zeros((), dtype=torch.float64, device=device)
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0], batch[1]
                h = model(x.to(device)).double()
                y_enc = _encode_targets(y.to(device), n_classes).double()
                preds = torch.einsum("kdc,bd->kbc", Ws, h) + biases.unsqueeze(1)
                res = preds - y_enc.unsqueeze(0)
                ss_res += (res**2).sum(dim=(1, 2))
                ss_tot_scalar += ((y_enc - y_bar_train) ** 2).sum()
        return 1.0 - ss_res / ss_tot_scalar.clamp(min=1e-12)

    # Classification: balanced accuracy via confusion matrix (K, C, C)
    confusion = torch.zeros(K, C, C, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0], batch[1]
            h = model(x.to(device)).double()
            y_true = y.to(device).long()
            preds = torch.einsum("kdc,bd->kbc", Ws, h) + biases.unsqueeze(1)
            y_pred = preds.argmax(dim=2)  # (K, B)
            # TODO: vectorize via scatter_add_ or per-k torch.bincount once val sets grow
            for k in range(K):
                for t, p in zip(y_true, y_pred[k]):
                    confusion[k, t.long(), p.long()] += 1

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
        verbose: int = 1,
    ):
        self.model_ = feature_extractor
        self.n_classes_ = n_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.lambdas = lambdas
        self.val_set = val_set
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

        W = self._result["W"]  # (D, C) float64
        bias = self._result["bias"]  # (C,) float64
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
                h = self.model_(x.to(self.device)).double()
                y_hat = h @ W + bias  # (B, C)
                outs.append(y_hat.cpu().numpy())
        preds = np.concatenate(outs, axis=0)

        if self.n_classes_ is None:
            return preds  # (N, C_out)
        return preds.argmax(axis=1).astype(np.int64)
