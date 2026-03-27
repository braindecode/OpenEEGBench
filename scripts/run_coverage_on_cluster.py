#!/usr/bin/env python
"""Launch minimal set of experiments to cover all required combinations.

Coverage targets:
- One successful run per dataset
- One successful run per backbone x finetuning
- One successful run per backbone x head
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
from open_eeg_bench.default_configs.backbones import ALL_BACKBONES
from open_eeg_bench.default_configs.datasets import ALL_DATASETS
from open_eeg_bench.default_configs import ALL_FINETUNING, ALL_HEADS
from open_eeg_bench.default_configs.experiments import default_training
from open_eeg_bench.head import LinearHead, MLPHead, OriginalHead
from open_eeg_bench.finetuning import Frozen, FullFinetune

RESULTS_DIR = Path("/expanse/projects/nemar/eeg_finetuning/pierre/oeb_results")

# Small, fast dataset for backbone x finetuning and backbone x head coverage
FAST_DATASET = "arithmetic_zyma2019"
# Datasets to skip (gated/inaccessible)
SKIP_DATASETS = set()


def build_coverage_experiments():
    """Build minimal set of experiments for full coverage."""
    experiments = {}  # (backbone, head_kind, finetuning_kind, dataset) -> Experiment

    def _key(bb, hk, fk, ds):
        return (bb, hk, fk, ds)

    def _add(bb_name, head, finetuning, ds_name):
        k = _key(bb_name, head.kind, finetuning.kind, ds_name)
        if k in experiments:
            return
        exp = Experiment.model_construct(
            training=default_training(),
            head=head,
            finetuning=finetuning,
            dataset=ALL_DATASETS[ds_name](),
            backbone=ALL_BACKBONES[bb_name](),
        )
        experiments[k] = exp

    # 1. backbone x finetuning: LinearHead(seed=1) + FAST_DATASET
    for bb_name in ALL_BACKBONES:
        for ft_name, ft_cls in ALL_FINETUNING.items():
            head = LinearHead(seed=1)
            ft = ft_cls()
            _add(bb_name, head, ft, FAST_DATASET)

    # 2. backbone x head: Frozen + FAST_DATASET (OriginalHead -> FullFinetune)
    for bb_name in ALL_BACKBONES:
        _add(bb_name, LinearHead(seed=1), Frozen(), FAST_DATASET)  # overlap
        _add(bb_name, MLPHead(seed=1), Frozen(), FAST_DATASET)
        _add(bb_name, OriginalHead(seed=1), FullFinetune(), FAST_DATASET)

    # 3. dataset coverage: BIOT + LinearHead(seed=1) + Frozen
    for ds_name in ALL_DATASETS:
        if ds_name in SKIP_DATASETS:
            continue
        _add("biot", LinearHead(seed=1), Frozen(), ds_name)

    return list(experiments.values())


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

    experiments = build_coverage_experiments()
    logging.info("Launching %d coverage experiments", len(experiments))

    infra_kwargs = dict(
        folder=str(RESULTS_DIR),
        cluster=args.cluster,
        min_samples_per_job=6,
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
