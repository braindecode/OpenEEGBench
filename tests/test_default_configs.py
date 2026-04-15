"""Test that all default configs instantiate and pass pydantic validation."""

import pytest

from open_eeg_bench.default_configs.backbones import ALL_BACKBONES
from open_eeg_bench.default_configs import ALL_FINETUNING, ALL_HEADS
from open_eeg_bench.default_configs.datasets import ALL_DATASETS
from open_eeg_bench.experiment import Experiment
from open_eeg_bench.dataset import PredefinedSplitter
from open_eeg_bench.finetuning import Frozen, FullFinetune
from open_eeg_bench.head import OriginalHead


@pytest.mark.parametrize("factory", ALL_BACKBONES.values(), ids=lambda f: f.__name__)
def test_backbone_instantiation(factory):
    cfg = factory()
    assert cfg.peft_target_modules  # non-empty


@pytest.mark.parametrize("cls", ALL_HEADS.values(), ids=lambda c: c.__name__)
def test_head_instantiation(cls):
    cfg = cls()
    assert cfg.kind


@pytest.mark.parametrize("cls", ALL_FINETUNING.values(), ids=lambda c: c.__name__)
def test_finetuning_instantiation(cls):
    cfg = cls()
    assert cfg.kind


@pytest.mark.parametrize("factory", ALL_DATASETS.values(), ids=lambda f: f.__name__)
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


def test_predefined_splitter_both_val_sources_rejected():
    """Providing both val_values and val_size should fail."""
    with pytest.raises(ValueError, match="Exactly one of val_values or val_size"):
        PredefinedSplitter(
            metadata_key="train",
            train_values=[True],
            val_values=[False],
            val_size=5,
            test_values=[False],
        )


def test_predefined_splitter_no_val_source_rejected():
    """Providing neither val_values nor val_size should fail."""
    with pytest.raises(ValueError, match="Exactly one of val_values or val_size"):
        PredefinedSplitter(
            metadata_key="train",
            train_values=[True],
            test_values=[False],
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


def test_training_has_sgd_kind():
    from open_eeg_bench.training import Training
    t = Training()
    assert t.kind == "sgd"


def test_ridge_probing_training_instantiation():
    from open_eeg_bench.training import RidgeProbingTraining
    t = RidgeProbingTraining()
    assert t.kind == "ridge"
    assert t.batch_size == 64
    assert t.lambdas is None
