from itertools import product
from typing import Iterable

from open_eeg_bench.experiment import Experiment
from open_eeg_bench.training import Training, EarlyStopping
from open_eeg_bench.default_configs import (
    ALL_DATASETS,
    ALL_HEADS,
    ALL_FINETUNING,
)
from open_eeg_bench.backbone import PlaceholderBackbone


def default_training():
    return Training(
        max_epochs=30,
        lr=1e-3,
        weight_decay=0.0,
        device="cuda",
        early_stopping=EarlyStopping(patience=10, monitor="valid_loss"),
    )


def make_all_experiments(
    datasets: list[str] | None = None,
    heads: list[str] | None = None,
    finetuning_strategies: list[str] | None = None,
    n_seeds: int = 3,
) -> list[Experiment]:
    """All  dataset x head x finetuning x seed combinations.
    Before running, replace the PlaceholderBackbone with actual backbones in the loop below.

    Parameters
    ----------
    datasets : list[str], optional
        Dataset names to evaluate on. If ``None``, uses all datasets.
        Valid names: ``"arithmetic_zyma2019"``, ``"bcic2a"``,
        ``"bcic2020_3"``, ``"physionet"``, ``"chbmit"``, ``"faced"``,
        ``"isruc_sleep"``, ``"mdd_mumtaz2016"``, ``"seed_v"``,
        ``"seed_vig"``, ``"tuab"``, ``"tuev"``.
    finetuning : list[str], optional
        Finetuning strategy names. If ``None``, uses ``["frozen"]``
        (linear probing). Valid names: ``"frozen"``, ``"lora"``,
        ``"ia3"``, ``"adalora"``, ``"dora"``, ``"oft"``,
        ``"full_finetune"``.
    heads : list[str], optional
        Head names to evaluate. If ``None``, uses ``["linear_head", "mlp_head"]``.
        Valid names: ``"linear_head"``, ``"mlp_head"``, ``"original_head"``.
    n_seeds : int
        Number of random seeds for initialization of the heads and new layers.
    """
    seeds = list(range(1, n_seeds + 1))
    heads = heads or ["linear_head", "mlp_head"]
    finetuning_strategies = finetuning_strategies or ["frozen"]
    datasets = datasets or list(ALL_DATASETS.keys())

    experiments = []
    for seed, head_name, finetuning_name, dataset_name in product(
        seeds, heads, finetuning_strategies, datasets
    ):
        exp = Experiment.model_construct(
            training=default_training(),
            head=ALL_HEADS[head_name](),
            finetuning=ALL_FINETUNING[finetuning_name](),
            dataset=ALL_DATASETS[dataset_name](),
            # Placeholder must be replaced by an actual backbone
            backbone=PlaceholderBackbone(),
            seed=seed,
        )
        experiments.append(exp)

    return experiments
