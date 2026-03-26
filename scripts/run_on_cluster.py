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
from open_eeg_bench.experiment import ExperimentHandler
from open_eeg_bench.default_configs.experiments import make_all_experiments
from open_eeg_bench.default_configs.backbones import ALL_BACKBONES

RESULTS_DIR = Path("/expanse/projects/nemar/eeg_finetuning/pierre/oeb_results")


def make_test_experiments():
    """Single BIOT + arithmetic_zyma2019 experiment for testing."""
    from open_eeg_bench.default_configs.backbones import biot

    experiments = make_all_experiments(
        datasets=["arithmetic_zyma2019"],
        heads=["linear_head"],
        finetuning_strategies=["frozen"],
    )
    experiments[0].backbone = biot()
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
        experiments_placeholder = make_all_experiments()
        experiments = []
        for exp in experiments_placeholder:
            for _, backbone_cls in ALL_BACKBONES.items():
                experiments.append(exp.model_copy(update={"backbone": backbone_cls()}))
    else:
        parser.error("Specify --test or --all")
        exit(1)

    logging.info(
        "Launching %d experiments (cluster=%s)", len(experiments), args.cluster
    )

    infra_kwargs = dict(
        folder=str(RESULTS_DIR),
        cluster=args.cluster,
        min_samples_per_job=6,
        mode="cached",
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
