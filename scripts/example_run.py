#!/usr/bin/env python
"""Example: run experiments with the TaskInfra-based Experiment class.

Demonstrates three modes:
1. Single experiment -- ``exp.run()``
2. Batch run locally -- ``run_many(experiments)`` with ``cluster=None``
3. Batch run on SLURM -- ``run_many(experiments)`` with ``cluster="slurm"``
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from open_eeg_bench.experiment import Experiment, run_many
from open_eeg_bench.default_configs.backbones import biot
from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019
from open_eeg_bench.head import LinearHead
from open_eeg_bench.finetuning import Frozen
from open_eeg_bench.training import Training, EarlyStopping

# -- Shared config ----------------------------------------------------------

backbone = biot()
dataset = arithmetic_zyma2019()
training = Training(
    max_epochs=30,
    lr=1e-3,
    weight_decay=0.0,
    device="cpu",
    early_stopping=EarlyStopping(patience=10, monitor="valid_loss"),
)


# ---------------------------------------------------------------------------
# 1. Single experiment
# ---------------------------------------------------------------------------
def single_run():
    exp = Experiment(
        seed=1,
        backbone=backbone,
        dataset=dataset,
        training=training,
        head=LinearHead(),
        finetuning=Frozen(),
        infra={"folder": "./results/single"},
    )
    result = exp.run()
    print("Single run result:", result)


# ---------------------------------------------------------------------------
# 2. Batch run -- local sequential (cluster=None is the default)
# ---------------------------------------------------------------------------
def batch_local():
    experiments = [
        Experiment(
            seed=seed,
            backbone=backbone,
            dataset=dataset,
            training=training,
            head=LinearHead(),
            finetuning=Frozen(),
            infra={"folder": "./results/batch_local"},
        )
        for seed in range(1, 4)
    ]
    df = run_many(experiments)
    print("Local batch results:\n", df)


# ---------------------------------------------------------------------------
# 3. Batch run -- SLURM job array
# ---------------------------------------------------------------------------
def batch_slurm():
    experiments = [
        Experiment(
            seed=seed,
            backbone=backbone,
            dataset=dataset,
            training=training,
            head=LinearHead(),
            finetuning=Frozen(),
            infra={
                "folder": "./results/batch_slurm",
                "cluster": "slurm",
                "gpus_per_node": 1,
                "mem_gb": 32,
                "timeout_min": 60,
            },
        )
        for seed in range(1, 4)
    ]
    # Submits a SLURM job array and returns immediately.
    # Results are collected from cache once the jobs complete.
    df = run_many(experiments)
    print("SLURM batch results:\n", df)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "single":
        single_run()
    elif mode == "local":
        batch_local()
    elif mode == "slurm":
        batch_slurm()
    else:
        print(f"Usage: {sys.argv[0]} [single|local|slurm]")
