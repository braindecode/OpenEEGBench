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
    datasets: Iterable[str] = ALL_DATASETS.keys(),
    heads: Iterable[str] = ALL_HEADS.keys(),
    finetuning_strategies: Iterable[str] = ALL_FINETUNING.keys(),
) -> list[Experiment]:
    """All  dataset x head x finetuning combinations.
    Before running, replace the PlaceholderBackbone with actual backbones in the loop below.
    """
    experiments = []
    for head_name, finetuning_name, dataset_name in product(
        heads, finetuning_strategies, datasets
    ):
        exp = Experiment.model_construct(
            training=default_training(),
            head=ALL_HEADS[head_name](),
            finetuning=ALL_FINETUNING[finetuning_name](),
            dataset=ALL_DATASETS[dataset_name](),
            # Placeholder must be replaced by an actual backbone
            backbone=PlaceholderBackbone(),
        )
        experiments.append(exp)

    return experiments
