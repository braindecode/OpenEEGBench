#!/usr/bin/env python
"""Launch one tuev and one tuab experiment on SLURM to validate val_size split.

Usage:
    python scripts/run_tuev_tuab_on_cluster.py
    python scripts/run_tuev_tuab_on_cluster.py --cluster local
"""

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault(
    "HF_HOME", "/expanse/projects/nemar/eeg_finetuning/pierre/hf_cache"
)

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
from open_eeg_bench.default_configs.backbones import biot
from open_eeg_bench.default_configs.datasets import tuev, tuab
from open_eeg_bench.default_configs.experiments import default_training
from open_eeg_bench.head import LinearHead
from open_eeg_bench.finetuning import Frozen

RESULTS_DIR = Path("/expanse/projects/nemar/eeg_finetuning/pierre/oeb_results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster", default="slurm", choices=["local", "slurm"],
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore cache and recompute all experiments",
    )
    args = parser.parse_args()

    backbone = biot()
    experiments = [
        Experiment.model_construct(
            training=default_training(),
            head=LinearHead(seed=1),
            finetuning=Frozen(),
            dataset=tuab(),
            backbone=backbone,
        ),
        Experiment.model_construct(
            training=default_training(),
            head=LinearHead(seed=1),
            finetuning=Frozen(),
            dataset=tuev(),
            backbone=backbone,
        ),
    ]

    logging.info("Launching %d experiments", len(experiments))
    for exp in experiments:
        logging.info("  - %s (val_size=%s)", exp.dataset.hf_id, exp.dataset.splitter.val_size)

    infra_kwargs = dict(
        folder=str(RESULTS_DIR),
        cluster=args.cluster,
        min_samples_per_job=1,
        mode="force" if args.force else "cached",
    )
    if args.cluster == "slurm":
        infra_kwargs.update(
            nodes=1,
            cpus_per_task=8,
            mem_gb=64,
            timeout_min=120,
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
