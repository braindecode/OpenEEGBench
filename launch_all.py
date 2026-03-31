"""Expanse SLURM launcher for OpenEEG-Bench.

Submits all backbone x dataset x seed combinations as a SLURM job array
on SDSC Expanse. Run from the login node with the conda env activated.

Usage:
    conda activate open-eeg-bench
    python launch_all.py
"""

import contextlib
import logging
import os
from pathlib import Path

# ── HuggingFace cache (must be set before any HF imports) ─────────────────
HF_HOME = "/expanse/projects/nemar/eeg_finetuning/pierre/hf_cache"
os.environ.setdefault("HF_HOME", HF_HOME)

# ── Expanse workaround: preserve SLURM_CONF across submitit's clean_env ──
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

# ── Expanse workaround: inject module loads into SLURM jobs ───────────────
import exca.slurm as _exca_slurm

_original_executor = _exca_slurm.SubmititMixin.executor


def _patched_executor(self):
    ex = _original_executor(self)
    if ex is not None:
        ex.update_parameters(
            slurm_setup=[
                "source ~/.bashrc",
                f"export HF_HOME={HF_HOME}",
                "module load gpu",
                "module load cuda12.2/toolkit/12.2.2",
            ]
        )
    return ex


_exca_slurm.SubmititMixin.executor = _patched_executor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

from open_eeg_bench.default_configs import ALL_BACKBONES
from open_eeg_bench.default_configs.experiments import make_all_experiments
from open_eeg_bench.experiment import run_many

# ── Configuration ──────────────────────────────────────────────────────────
BACKBONE_NAMES = ["cbramod", "labram", "reve", "eegnet"]
DATASET_NAMES = ["faced"]
FINETUNING_NAMES = ["full_finetune"]
HEAD_NAMES = ["linear_head"]
N_SEEDS = 2
SANITY_CHECK = False
RESULTS_FOLDER = Path("/expanse/projects/nemar/bruno/cache/exca/")

# ── Build experiment grid ──────────────────────────────────────────────────
backbones = [ALL_BACKBONES[name]() for name in BACKBONE_NAMES]

experiments_template = make_all_experiments(
    datasets=DATASET_NAMES,
    heads=HEAD_NAMES,
    finetuning_strategies=FINETUNING_NAMES,
    n_seeds=N_SEEDS,
)

# Replace the placeholder backbone with each real backbone
all_experiments = []
for backbone in backbones:
    training_overrides = {"device": "cuda", "max_epochs": 100}
    if SANITY_CHECK:
        training_overrides.update({"max_epochs": 1, "early_stopping": {"enabled": False}})
    overrides = {
        "backbone": backbone,
        "training": training_overrides,
        "infra": {
            "folder": str(RESULTS_FOLDER),
            "cluster": "slurm",
            "slurm_partition": "gpu-shared",
            "slurm_account": "csd403",
            "timeout_min": 180,
            "nodes": 1,
            "mem_gb": 32,
            "cpus_per_task": 8,
            "slurm_additional_parameters": {
                "gpus": 1,
                "qos": "gpu-shared-normal",
            },
        },
    }
    all_experiments.extend(
        [exp.infra.clone_obj(overrides) for exp in experiments_template]
    )

log.info(
    "Launching %d experiments: %d backbones x %d datasets x %d seeds",
    len(all_experiments),
    len(BACKBONE_NAMES),
    len(DATASET_NAMES),
    N_SEEDS,
)

# ── Submit as SLURM job array ─────────────────────────────────────────────
results = run_many(all_experiments, max_workers=20)

if not results.empty:
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(results.to_string())
    csv_path = RESULTS_FOLDER / "results.csv"
    results.to_csv(csv_path, index=False)
    log.info("Results saved to %s", csv_path)
else:
    log.warning("No completed experiments to report.")
