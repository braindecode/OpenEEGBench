from itertools import product

from open_eeg_bench.experiment import Experiment
from open_eeg_bench.training import Training, RidgeProbingTraining, EarlyStopping
from open_eeg_bench.finetuning import Frozen
from open_eeg_bench.head import FlattenHead
from open_eeg_bench.default_configs import (
    ALL_DATASETS,
    ALL_HEADS,
    ALL_FINETUNING,
)
from open_eeg_bench.backbone import PlaceholderBackbone


RIDGE_PROBE_NAME = "ridge_probe"


def default_training() -> Training:
    return Training(
        max_epochs=30,
        lr=1e-3,
        device="cpu",
        early_stopping=EarlyStopping(patience=10, monitor="valid_loss"),
    )


def default_ridge_probing() -> RidgeProbingTraining:
    return RidgeProbingTraining(device="cpu")


def _valid_strategies() -> set[str]:
    return set(ALL_FINETUNING.keys()) | {RIDGE_PROBE_NAME}


def make_all_experiments(
    datasets: list[str] | None = None,
    heads: list[str] | None = None,
    finetuning_strategies: list[str] | None = None,
    n_seeds: int = 3,
) -> list[Experiment]:
    """All dataset × head × finetuning × seed combinations.

    Before running, replace the PlaceholderBackbone with actual backbones.

    Parameters
    ----------
    datasets : list[str], optional
        Dataset names to evaluate on. If None, uses all datasets.
    finetuning_strategies : list[str], optional
        Finetuning strategy names. If None, uses ["frozen"].
        Valid: "frozen", "lora", "ia3", "adalora", "dora", "oft",
        "full_finetune", "two_stages", "ridge_probe".
        "ridge_probe" performs closed-form ridge regression linear probing.
    heads : list[str], optional
        Head names. If None, uses ["linear_head"]. Ignored for "ridge_probe"
        (which always uses FlattenHead by definition).
    n_seeds : int
        Number of random seeds.
    """
    seeds = list(range(n_seeds))
    heads = heads or ["linear_head"]
    finetuning_strategies = finetuning_strategies or ["frozen"]
    datasets = datasets or list(ALL_DATASETS.keys())

    unknown = set(finetuning_strategies) - _valid_strategies()
    if unknown:
        raise ValueError(f"Unknown finetuning strategies: {sorted(unknown)}")

    experiments = []
    for seed, head_name, finetuning_name, dataset_name in product(
        seeds, heads, finetuning_strategies, datasets
    ):
        if finetuning_name == RIDGE_PROBE_NAME:
            # Ridge probe is linear and head-agnostic — deduplicate across heads
            if head_name != heads[0]:
                continue
            head_cfg = FlattenHead()
            training_cfg = default_ridge_probing()
            finetuning_cfg = Frozen()
        else:
            head_cfg = ALL_HEADS[head_name]()
            training_cfg = default_training()
            finetuning_cfg = ALL_FINETUNING[finetuning_name]()

        exp = Experiment(
            training=training_cfg,
            head=head_cfg,
            finetuning=finetuning_cfg,
            dataset=ALL_DATASETS[dataset_name](),
            backbone=PlaceholderBackbone(),
            seed=seed,
        )
        experiments.append(exp)

    return experiments
