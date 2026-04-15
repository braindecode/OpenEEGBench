"""Streaming ridge regression probe for frozen backbone evaluation.

Accumulates sufficient statistics (X^TX, X^TY, centering sums) in float64
over a single pass, eigendecomposes the centered covariance once, then
sweeps λ cheaply in the eigenbasis for each val/test pass.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _encode_targets(y: torch.Tensor, n_classes: int | None) -> torch.Tensor:
    """One-hot encode class labels, or reshape regression targets to 2D."""
    if n_classes is None:
        # Regression: ensure (B, C) shape
        return y.float().reshape(y.shape[0], -1)
    # Classification: one-hot
    return torch.nn.functional.one_hot(y.long(), num_classes=n_classes).float()


def _default_lambdas(eigvals_mean: float) -> list[float]:
    """Default λ grid: 9 values on a logspace scaled by mean eigenvalue."""
    scale = max(eigvals_mean, 1e-12)
    return [10**e * scale for e in range(-4, 5)]


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
    A = B = s_h = s_y = None
    N = 0
    with torch.no_grad():
        for x, y in train_loader:
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
                s_y = torch.zeros(C, dtype=torch.float64, device=device)

            A += h64.T @ h64
            B += h64.T @ y64
            s_h += h64.sum(0)
            s_y += y64.sum(0)
            N += h64.shape[0]

    if A is None:
        raise ValueError("Empty train_loader — no features accumulated.")

    D = A.shape[0]
    C = B.shape[1]

    # ----- Center (bias handled analytically) -----
    h_bar = s_h / N
    y_bar = s_y / N
    C_xx = A - N * torch.outer(h_bar, h_bar)
    C_xy = B - N * torch.outer(h_bar, y_bar)

    # ----- Eigendecomposition (once) -----
    eigvals, Q = torch.linalg.eigh(C_xx)  # C_xx symmetric PSD
    Ct = Q.T @ C_xy                         # (D, C)

    if lambdas is None:
        lambdas = _default_lambdas(eigvals.mean().item())

    # ----- Solve for each λ in eigenbasis -----
    K = len(lambdas)
    Ws = torch.zeros(K, D, C, dtype=torch.float64, device=device)
    biases = torch.zeros(K, C, dtype=torch.float64, device=device)
    for k, lam in enumerate(lambdas):
        denom = (eigvals + lam).unsqueeze(1)  # (D, 1)
        Ws[k] = Q @ (Ct / denom)               # (D, C)
        biases[k] = y_bar - Ws[k].T @ h_bar    # (C,)

    # ----- Pass 2: streaming λ selection on val -----
    val_scores = _streaming_val_scores(
        model, val_loader, Ws, biases, n_classes=n_classes,
        y_bar_train=y_bar, device=device,
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
    Ws: torch.Tensor,       # (K, D, C)
    biases: torch.Tensor,   # (K, C)
    n_classes: int | None,
    y_bar_train: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Accumulate per-λ metric streaming on val. Returns (K,) scores (higher=better)."""
    K, _, C = Ws.shape
    is_regression = n_classes is None

    if is_regression:
        # R² = 1 - SS_res / SS_tot  (SS_tot using train mean; matches sklearn behavior closely enough)
        ss_res = torch.zeros(K, dtype=torch.float64, device=device)
        ss_tot = torch.zeros(K, dtype=torch.float64, device=device)
        with torch.no_grad():
            for x, y in val_loader:
                h = model(x.to(device)).double()
                y_enc = _encode_targets(y.to(device), n_classes).double()
                preds = torch.einsum('kdc,bd->kbc', Ws, h) + biases.unsqueeze(1)
                res = preds - y_enc.unsqueeze(0)
                ss_res += (res ** 2).sum(dim=(1, 2))
                tot = y_enc - y_bar_train.unsqueeze(0)
                ss_tot += (tot ** 2).sum() * torch.ones(K, dtype=torch.float64, device=device)
        return 1.0 - ss_res / ss_tot.clamp(min=1e-12)

    # Classification: balanced accuracy via confusion matrix (K, C, C)
    confusion = torch.zeros(K, C, C, dtype=torch.float64, device=device)
    with torch.no_grad():
        for x, y in val_loader:
            h = model(x.to(device)).double()
            y_true = y.to(device).long()
            preds = torch.einsum('kdc,bd->kbc', Ws, h) + biases.unsqueeze(1)
            y_pred = preds.argmax(dim=2)  # (K, B)
            for k in range(K):
                for t, p in zip(y_true, y_pred[k]):
                    confusion[k, t.long(), p.long()] += 1

    # Balanced accuracy = mean of per-class recall
    per_class = confusion.sum(dim=2).clamp(min=1)
    recalls = confusion.diagonal(dim1=1, dim2=2) / per_class
    return recalls.mean(dim=1)
