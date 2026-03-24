#!/usr/bin/env python
"""Launch experiments on SLURM via ExperimentHandler.

Usage:
    # Single test run
    python scripts/run_on_cluster.py --test

    # All combinations
    python scripts/run_on_cluster.py --all
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from exca import MapInfra

from open_eeg_bench.experiment import Experiment, ExperimentHandler
from open_eeg_bench.default_configs.backbones import ALL_BACKBONES
from open_eeg_bench.default_configs.datasets import ALL_DATASETS
from open_eeg_bench.default_configs.heads import linear_head
from open_eeg_bench.finetuning import Frozen
from open_eeg_bench.training import Training, EarlyStopping

RESULTS_DIR = Path("/expanse/projects/nemar/eeg_finetuning/pierre/oeb_results")
HF_CACHE = "/expanse/projects/nemar/eeg_finetuning/pierre/hf_cache"

TRAINING = Training(
    max_epochs=30,
    lr=1e-3,
    weight_decay=0.0,
    device="cuda",
    early_stopping=EarlyStopping(patience=10, monitor="valid_loss"),
)


def make_test_experiments():
    """Single BIOT + arithmetic_zyma2019 experiment for testing."""
    from open_eeg_bench.default_configs.backbones import biot
    from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019

    return [
        Experiment(
            backbone=biot(),
            head=linear_head(),
            finetuning=Frozen(),
            dataset=arithmetic_zyma2019(),
            training=TRAINING,
        )
    ]


def make_all_experiments():
    """All backbone × dataset × linear head combinations with frozen encoder."""
    experiments = []
    for backbone_fn in ALL_BACKBONES:
        for dataset_fn in ALL_DATASETS:
            try:
                exp = Experiment(
                    backbone=backbone_fn(),
                    head=linear_head(),
                    finetuning=Frozen(),
                    dataset=dataset_fn(),
                    training=TRAINING,
                )
                experiments.append(exp)
            except Exception as e:
                logging.warning(
                    "Skipping %s × %s: %s",
                    backbone_fn.__name__,
                    dataset_fn.__name__,
                    e,
                )
    return experiments


def main():
    import os
    os.environ.setdefault("HF_HOME", HF_CACHE)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Single test experiment")
    parser.add_argument("--all", action="store_true", help="All combinations")
    args = parser.parse_args()

    if args.test:
        experiments = make_test_experiments()
    elif args.all:
        experiments = make_all_experiments()
    else:
        parser.error("Specify --test or --all")

    logging.info("Launching %d experiments", len(experiments))

    handler = ExperimentHandler(
        parallelise_within_node=False,
        infra=MapInfra(
            folder=str(RESULTS_DIR),
            cluster="slurm",
            gpus_per_node=1,
            cpus_per_task=8,
            mem_gb=32,
            timeout_min=60,
            slurm_partition="gpu-shared",
            slurm_account="csd403",
            min_samples_per_job=1,
        ),
    )

    results = list(handler.run(experiments))
    logging.info("Completed %d experiments", len(results))
    for r in results:
        print(r.to_string(index=False))


if __name__ == "__main__":
    main()
