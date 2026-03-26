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
import os
from pathlib import Path

os.environ.setdefault(
    "HF_HOME", "/expanse/projects/nemar/eeg_finetuning/pierre/hf_cache"
)

# Workaround: exca calls submitit.helpers.clean_env() which removes all
# SLURM_* env vars before calling sbatch. On Expanse, SLURM_CONF is required
# for sbatch to find its config. We monkey-patch clean_env to preserve it.
import contextlib
import submitit.helpers

_original_clean_env = submitit.helpers.clean_env

@contextlib.contextmanager
def _clean_env_preserve_conf(*args, **kwargs):
    slurm_conf = os.environ.get("SLURM_CONF")
    with _original_clean_env(*args, **kwargs):
        if slurm_conf is not None:
            os.environ["SLURM_CONF"] = slurm_conf
        yield

submitit.helpers.clean_env = _clean_env_preserve_conf

# Monkey-patch exca's SubmititMixin.executor() to add module load commands.
# Expanse requires loading CUDA modules for GPU jobs.
import exca.slurm as _exca_slurm

_original_executor = _exca_slurm.SubmititMixin.executor

def _patched_executor(self):
    ex = _original_executor(self)
    if ex is not None:
        ex.update_parameters(
            slurm_setup=[
                "source ~/.bashrc",
                "module load gpu",
                "module load cuda12.2/toolkit/12.2.2",
            ]
        )
    return ex

_exca_slurm.SubmititMixin.executor = _patched_executor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from exca import MapInfra

from open_eeg_bench.experiment import Experiment, ExperimentHandler
from open_eeg_bench.default_configs.backbones import ALL_BACKBONES
from open_eeg_bench.default_configs.datasets import (
    arithmetic_zyma2019,
    bcic2a,
    bcic2020_3,
    physionet,
    chbmit,
    faced,
    isruc_sleep,
    mdd_mumtaz2016,
    seed_v,
)

# Datasets already downloaded on the cluster (skip seed-vig, tuab, tuev for now)
AVAILABLE_DATASETS = [
    arithmetic_zyma2019,
    bcic2a,
    bcic2020_3,
    physionet,
    chbmit,
    faced,
    isruc_sleep,
    mdd_mumtaz2016,
    seed_v,
]
from open_eeg_bench.default_configs.heads import linear_head
from open_eeg_bench.finetuning import Frozen
from open_eeg_bench.training import Training, EarlyStopping

RESULTS_DIR = Path("/expanse/projects/nemar/eeg_finetuning/pierre/oeb_results")

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
    """All backbone x dataset x linear head combinations with frozen encoder."""
    experiments = []
    for backbone_fn in ALL_BACKBONES:
        for dataset_fn in AVAILABLE_DATASETS:
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
                    "Skipping %s x %s: %s",
                    backbone_fn.__name__,
                    dataset_fn.__name__,
                    e,
                )
    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Single test experiment")
    parser.add_argument("--all", action="store_true", help="All combinations")
    parser.add_argument(
        "--cluster",
        default="local",
        choices=["local", "slurm"],
        help="Execution backend (default: local, for running inside a SLURM job)",
    )
    args = parser.parse_args()

    if args.test:
        experiments = make_test_experiments()
    elif args.all:
        experiments = make_all_experiments()
    else:
        parser.error("Specify --test or --all")

    logging.info(
        "Launching %d experiments (cluster=%s)", len(experiments), args.cluster
    )

    infra_kwargs = dict(
        folder=str(RESULTS_DIR),
        cluster=args.cluster,
        min_samples_per_job=6,
        mode="force",  # recompute everything (clear corrupted cache from OOM jobs)
    )
    if args.cluster == "slurm":
        infra_kwargs.update(
            nodes=1,
            cpus_per_task=8,
            mem_gb=64,
            timeout_min=60,
            slurm_partition="gpu-shared",
            slurm_account="csd403",
            slurm_additional_parameters={
                "gpus": 1,
                "exclude": "exp-1-57",
                "qos": "gpu-shared-normal",
            },
        )

    handler = ExperimentHandler(
        parallelise_within_node=False,
        infra=MapInfra(**infra_kwargs),
    )

    results = list(handler.run(experiments))
    logging.info("Completed %d experiments", len(results))
    for r in results:
        print(r.to_string(index=False))


if __name__ == "__main__":
    main()
