#!/usr/bin/env python
"""Launch a random sample of experiments on SLURM for validation.

Usage:
    python scripts/run_sample_on_cluster.py --n 50
"""

import argparse
import logging
import os
import random
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
from open_eeg_bench.experiment import ExperimentHandler
from open_eeg_bench.default_configs.experiments import make_all_experiments
from open_eeg_bench.default_configs.backbones import ALL_BACKBONES

RESULTS_DIR = Path("/expanse/projects/nemar/eeg_finetuning/pierre/oeb_results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of experiments to sample")
    parser.add_argument(
        "--cluster",
        default="slurm",
        choices=["local", "slurm"],
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    args = parser.parse_args()

    # Build all experiments with real backbones
    experiments_placeholder = make_all_experiments()
    all_experiments = []
    for exp in experiments_placeholder:
        for _, backbone_cls in ALL_BACKBONES.items():
            all_experiments.append(exp.model_copy(update={"backbone": backbone_cls()}))

    logging.info("Total experiments: %d", len(all_experiments))

    # Sample
    rng = random.Random(args.seed)
    sample = rng.sample(all_experiments, min(args.n, len(all_experiments)))
    logging.info("Sampled %d experiments", len(sample))

    for i, exp in enumerate(sample):
        logging.info(
            "  [%d] backbone=%s head=%s(seed=%s) finetuning=%s dataset=%s",
            i, type(exp.backbone).__name__,
            type(exp.head).__name__, getattr(exp.head, 'seed', '?'),
            type(exp.finetuning).__name__,
            type(exp.dataset).__name__,
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

    results = list(handler.run(sample))
    logging.info("Completed %d experiments", len(results))
    for r in results:
        print(r.to_string(index=False))


if __name__ == "__main__":
    main()
