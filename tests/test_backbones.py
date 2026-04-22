"""Test that backbone models can be built and produce correct output shapes."""

import pytest
import torch
import mne
from huggingface_hub.errors import GatedRepoError

from open_eeg_bench.backbone import ScratchBackbone
from open_eeg_bench.default_configs.backbones import ALL_BACKBONES
from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019
from open_eeg_bench.experiment import Experiment
from open_eeg_bench.finetuning import FullFinetune, Frozen, LoRA

# Use standard 10-20 channel names that all models recognize
CH_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]
N_CHANS = len(CH_NAMES)
N_TIMES = 1000
N_OUTPUTS = 2
SFREQ = 200.0


def _make_chs_info() -> list:
    """Create MNE-compatible chs_info with channel positions."""
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    info.set_montage(montage)
    return info["chs"]


@pytest.fixture(scope="module")
def chs_info():
    return _make_chs_info()


def _build(factory, chs_info):
    """Build a backbone, skipping if its weights are in a gated HF repo."""
    backbone = factory()
    try:
        model = backbone.build(
            n_chans=N_CHANS,
            n_times=N_TIMES,
            n_outputs=N_OUTPUTS,
            sfreq=SFREQ,
            chs_info=chs_info,
        )
    except GatedRepoError as e:
        pytest.skip(f"Gated HF repo (set HF_TOKEN to enable): {e}")
    return backbone, model


@pytest.mark.parametrize("factory", ALL_BACKBONES.values(), ids=lambda f: f.__name__)
def test_backbone_build_and_forward(factory, chs_info):
    """Build each backbone and verify forward pass runs without error."""
    _, model = _build(factory, chs_info)

    x = torch.randn(2, N_CHANS, N_TIMES)
    with torch.no_grad():
        model.eval()
        out = model(x)

    assert out.shape[0] == 2, f"Batch dim mismatch: {out.shape}"


@pytest.mark.parametrize("factory", ALL_BACKBONES.values(), ids=lambda f: f.__name__)
def test_backbone_has_head_module(factory, chs_info):
    """Verify each backbone has the declared head_module_name attribute."""
    backbone, model = _build(factory, chs_info)
    assert hasattr(model, backbone.head_module_name), (
        f"Model has no attribute '{backbone.head_module_name}'. "
        f"Available: {[n for n, _ in model.named_children()]}"
    )


def _eegnet_scratch() -> ScratchBackbone:
    return ScratchBackbone(
        model_cls="braindecode.models.EEGNet",
        peft_target_modules=None,
        head_module_name="final_layer",
    )


def test_scratch_backbone_build_and_forward(chs_info):
    """ScratchBackbone instantiates an EEGNet model with random weights."""
    backbone = _eegnet_scratch()
    model = backbone.build(
        n_chans=N_CHANS,
        n_times=N_TIMES,
        n_outputs=N_OUTPUTS,
        sfreq=SFREQ,
        chs_info=chs_info,
    )
    x = torch.randn(2, N_CHANS, N_TIMES)
    with torch.no_grad():
        model.eval()
        out = model(x)
    assert out.shape[0] == 2
    assert hasattr(model, backbone.head_module_name)


def test_scratch_backbone_requires_full_finetune():
    """ScratchBackbone with anything other than FullFinetune must be rejected."""
    with pytest.raises(ValueError, match="ScratchBackbone.*FullFinetune"):
        Experiment(
            backbone=_eegnet_scratch(),
            finetuning=Frozen(),
            dataset=arithmetic_zyma2019(),
        )
    with pytest.raises(ValueError, match="ScratchBackbone.*FullFinetune"):
        Experiment(
            backbone=_eegnet_scratch(),
            finetuning=LoRA(),
            dataset=arithmetic_zyma2019(),
        )


def test_scratch_backbone_with_full_finetune_accepted():
    """ScratchBackbone + FullFinetune is a valid combination."""
    exp = Experiment(
        backbone=_eegnet_scratch(),
        finetuning=FullFinetune(),
        dataset=arithmetic_zyma2019(),
    )
    assert exp.backbone.kind == "scratch"
    assert exp.finetuning.kind == "full"
