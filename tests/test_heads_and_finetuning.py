"""Test head application and finetuning strategies on a real backbone."""

import pytest
import torch
import torch.nn as nn

from open_eeg_bench.default_configs.backbones import biot
from open_eeg_bench.head import LinearHead, MLPHead, OriginalHead
from open_eeg_bench.finetuning import LoRA, IA3, Frozen, FullFinetune


N_CHANS = 16
N_TIMES = 1000
N_OUTPUTS = 2
SFREQ = 200.0


def _build_biot():
    backbone = biot(hub_repo=None, checkpoint_url=None)
    model = backbone.build(
        n_chans=N_CHANS, n_times=N_TIMES, n_outputs=N_OUTPUTS, sfreq=SFREQ,
    )
    return backbone, model


def _init_lazy(model):
    """Initialize lazy modules with dummy forward."""
    has_lazy = any(isinstance(p, nn.parameter.UninitializedParameter) for p in model.parameters())
    if has_lazy:
        with torch.no_grad():
            model.eval()
            model(torch.zeros(1, N_CHANS, N_TIMES))
            model.train()


class TestHeads:
    def test_linear_head(self):
        backbone, model = _build_biot()
        LinearHead().apply(model, N_OUTPUTS, backbone.head_module_name)
        _init_lazy(model)
        out = model(torch.randn(2, N_CHANS, N_TIMES))
        assert out.shape == (2, N_OUTPUTS)

    def test_mlp_head(self):
        backbone, model = _build_biot()
        MLPHead(hidden_dim=64).apply(model, N_OUTPUTS, backbone.head_module_name)
        _init_lazy(model)
        out = model(torch.randn(2, N_CHANS, N_TIMES))
        assert out.shape == (2, N_OUTPUTS)

    def test_original_head(self):
        backbone, model = _build_biot()
        OriginalHead().apply(model, N_OUTPUTS, backbone.head_module_name)
        out = model(torch.randn(2, N_CHANS, N_TIMES))
        assert out.shape == (2, N_OUTPUTS)


class TestFinetuning:
    def test_frozen(self):
        backbone, model = _build_biot()
        LinearHead().apply(model, N_OUTPUTS, backbone.head_module_name)
        _init_lazy(model)
        model, stats = Frozen().apply(model, backbone)
        # Head should be trainable
        head = getattr(model, backbone.head_module_name)
        trainable_in_head = sum(p.requires_grad for p in head.parameters())
        assert trainable_in_head > 0
        assert stats["trainable_pct"] < 50  # most params frozen

    def test_full_finetune(self):
        backbone, model = _build_biot()
        model, stats = FullFinetune().apply(model, backbone)
        assert stats["trainable_pct"] > 99.0  # some buffers may not be trainable

    def test_lora(self):
        backbone, model = _build_biot()
        LinearHead().apply(model, N_OUTPUTS, backbone.head_module_name)
        _init_lazy(model)
        model, stats = LoRA(r=4, alpha=8).apply(model, backbone)
        assert stats["method"] == "lora"
        assert 0 < stats["trainable_pct"] < 100

    def test_ia3(self):
        backbone, model = _build_biot()
        LinearHead().apply(model, N_OUTPUTS, backbone.head_module_name)
        _init_lazy(model)
        model, stats = IA3().apply(model, backbone)
        assert stats["method"] == "ia3"
        assert 0 < stats["trainable_pct"] < 100
