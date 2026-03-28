"""Run a benchmark on a SLURM cluster.

Launch this script from the login node. It submits experiments as a
SLURM job array and returns immediately. Re-run the same script later
to collect completed results.

Adapt slurm_partition and slurm_account to your cluster.
"""

from open_eeg_bench import benchmark
from open_eeg_bench.default_configs.backbones import biot

backbone = biot()

results = benchmark(
    model_cls=backbone.model_cls,
    hub_repo=backbone.hub_repo,
    model_kwargs=backbone.model_kwargs,
    peft_target_modules=backbone.peft_target_modules,
    peft_ff_modules=backbone.peft_ff_modules,
    normalization=backbone.normalization,
    datasets=["arithmetic_zyma2019"],
    n_seeds=1,
    device="cuda",
    infra={
        "folder": "./results",
        "cluster": "slurm",
        "gpus_per_node": 1,
        "mem_gb": 32,
        "timeout_min": 120,
        "cpus_per_task": 4,
        "slurm_partition": "gpu",       # adapt to your cluster
        "slurm_account": "my_account",  # adapt to your cluster
    },
    max_workers=16,
)

print(results)
