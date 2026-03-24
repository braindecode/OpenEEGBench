"""Test that backbone models can be built and produce correct output shapes."""

import pytest
import torch
import mne

from open_eeg_bench.default_configs.backbones import ALL_BACKBONES

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


@pytest.mark.parametrize("factory", ALL_BACKBONES, ids=lambda f: f.__name__)
def test_backbone_build_and_forward(factory, chs_info):
    """Build each backbone and verify forward pass runs without error."""
    backbone = factory(hub_repo=None, checkpoint_url=None)

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

    assert out.shape[0] == 2, f"Batch dim mismatch: {out.shape}"


@pytest.mark.parametrize("factory", ALL_BACKBONES, ids=lambda f: f.__name__)
def test_backbone_has_head_module(factory, chs_info):
    """Verify each backbone has the declared head_module_name attribute."""
    backbone = factory(hub_repo=None, checkpoint_url=None)
    model = backbone.build(
        n_chans=N_CHANS,
        n_times=N_TIMES,
        n_outputs=N_OUTPUTS,
        sfreq=SFREQ,
        chs_info=chs_info,
    )
    assert hasattr(model, backbone.head_module_name), (
        f"Model has no attribute '{backbone.head_module_name}'. "
        f"Available: {[n for n, _ in model.named_children()]}"
    )
