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


def test_ridge_requires_flatten_head():
    """Ridge training without FlattenHead must be rejected."""
    from open_eeg_bench.default_configs.backbones import biot
    from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019
    from open_eeg_bench.training import RidgeProbingTraining
    from open_eeg_bench.head import LinearHead

    with pytest.raises(ValueError, match="FlattenHead"):
        Experiment(
            backbone=biot(),
            head=LinearHead(),
            finetuning=Frozen(),
            dataset=arithmetic_zyma2019(),
            training=RidgeProbingTraining(),
        )


def test_flatten_head_requires_ridge():
    """FlattenHead outside ridge training must be rejected."""
    from open_eeg_bench.default_configs.backbones import biot
    from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019
    from open_eeg_bench.head import FlattenHead
    from open_eeg_bench.training import Training

    with pytest.raises(ValueError, match="FlattenHead"):
        Experiment(
            backbone=biot(),
            head=FlattenHead(),
            finetuning=Frozen(),
            dataset=arithmetic_zyma2019(),
            training=Training(),
        )


def test_ridge_requires_frozen():
    """Ridge training with non-Frozen finetuning must be rejected."""
    from open_eeg_bench.default_configs.backbones import biot
    from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019
    from open_eeg_bench.finetuning import LoRA
    from open_eeg_bench.head import FlattenHead
    from open_eeg_bench.training import RidgeProbingTraining

    with pytest.raises(ValueError, match="Frozen"):
        Experiment(
            backbone=biot(),
            head=FlattenHead(),
            finetuning=LoRA(),
            dataset=arithmetic_zyma2019(),
            training=RidgeProbingTraining(),
        )


def test_ridge_rejects_training_required_modules():
    """Backbones with training_required_modules are incompatible with ridge."""
    from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019
    from open_eeg_bench.default_configs.backbones import biot
    from open_eeg_bench.head import FlattenHead
    from open_eeg_bench.training import RidgeProbingTraining

    bb = biot().model_copy(update={"training_required_modules": ["some_module"]})
    with pytest.raises(ValueError, match="training_required_modules"):
        Experiment(
            backbone=bb,
            head=FlattenHead(),
            finetuning=Frozen(),
            dataset=arithmetic_zyma2019(),
            training=RidgeProbingTraining(),
        )


def test_ridge_valid_combination_accepted():
    """ridge + FlattenHead + Frozen + clean backbone must be accepted."""
    from open_eeg_bench.default_configs.backbones import biot
    from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019
    from open_eeg_bench.head import FlattenHead
    from open_eeg_bench.training import RidgeProbingTraining

    exp = Experiment(
        backbone=biot(),
        head=FlattenHead(),
        finetuning=Frozen(),
        dataset=arithmetic_zyma2019(),
        training=RidgeProbingTraining(),
    )
    assert exp.training.kind == "ridge"
    assert exp.head.kind == "flatten"


def test_make_all_experiments_ridge_probe_no_duplicates():
    """ridge_probe strategy ignores `heads` and generates n_seeds × n_datasets experiments."""
    from open_eeg_bench.default_configs.experiments import make_all_experiments

    exps = make_all_experiments(
        datasets=["arithmetic_zyma2019"],
        heads=["linear_head", "mlp_head"],      # should be ignored for ridge_probe
        finetuning_strategies=["ridge_probe"],
        n_seeds=3,
    )
    assert len(exps) == 3   # 3 seeds × 1 dataset, heads ignored
    for e in exps:
        assert e.training.kind == "ridge"
        assert e.head.kind == "flatten"
        assert e.finetuning.kind == "frozen"


def test_make_all_experiments_mixed_strategies():
    """ridge_probe + frozen generate distinct experiments."""
    from open_eeg_bench.default_configs.experiments import make_all_experiments

    exps = make_all_experiments(
        datasets=["arithmetic_zyma2019"],
        heads=["linear_head"],
        finetuning_strategies=["frozen", "ridge_probe"],
        n_seeds=1,
    )
    assert len(exps) == 2
    kinds = sorted(e.training.kind for e in exps)
    assert kinds == ["ridge", "sgd"]
