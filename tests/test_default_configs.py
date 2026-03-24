"""Test that all default configs instantiate and pass pydantic validation."""

import pytest

from open_eeg_bench.default_configs.backbones import ALL_BACKBONES
from open_eeg_bench.default_configs.finetuning import ALL_FINETUNING
from open_eeg_bench.default_configs.heads import ALL_HEADS
from open_eeg_bench.default_configs.datasets import ALL_DATASETS
from open_eeg_bench.experiment import Experiment
from open_eeg_bench.finetuning import Frozen, FullFinetune
from open_eeg_bench.head import OriginalHead


@pytest.mark.parametrize("factory", ALL_BACKBONES, ids=lambda f: f.__name__)
def test_backbone_instantiation(factory):
    cfg = factory()
    assert cfg.peft_target_modules  # non-empty


@pytest.mark.parametrize("factory", ALL_HEADS, ids=lambda f: f.__name__)
def test_head_instantiation(factory):
    cfg = factory()
    assert cfg.kind


@pytest.mark.parametrize("factory", ALL_FINETUNING, ids=lambda f: f.__name__)
def test_finetuning_instantiation(factory):
    cfg = factory()
    assert cfg.kind


@pytest.mark.parametrize("factory", ALL_DATASETS, ids=lambda f: f.__name__)
def test_dataset_instantiation(factory):
    cfg = factory()
    assert cfg.hf_id


def test_experiment_instantiation():
    """Minimal experiment config validates successfully."""
    from open_eeg_bench.default_configs.backbones import biot
    from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019

    exp = Experiment(
        backbone=biot(),
        dataset=arithmetic_zyma2019(),
    )
    assert exp.seed == 42


def test_frozen_with_original_head_rejected():
    """Frozen encoder + original head should fail validation."""
    from open_eeg_bench.default_configs.backbones import biot
    from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019

    with pytest.raises(ValueError, match="Frozen.*OriginalHead"):
        Experiment(
            backbone=biot(),
            head=OriginalHead(),
            finetuning=Frozen(),
            dataset=arithmetic_zyma2019(),
        )


def test_full_finetune_with_original_head_accepted():
    """Full finetune + original head should be valid."""
    from open_eeg_bench.default_configs.backbones import biot
    from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019

    exp = Experiment(
        backbone=biot(),
        head=OriginalHead(),
        finetuning=FullFinetune(),
        dataset=arithmetic_zyma2019(),
    )
    assert exp.finetuning.kind == "full"
