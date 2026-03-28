# Running on a SLURM cluster

Open EEG Bench integrates with SLURM natively through the `infra` parameter — **you do not need to write any `sbatch` or `srun` script**. The benchmark function handles job submission, caching, and result collection for you.

All heavy imports (PyTorch, NumPy, braindecode, etc.) are **lazy**: they only happen inside the submitted jobs, not at script launch time. This means your launcher script is safe to run directly on a login node.

## 1. Write a launcher script

Create a short Python script (e.g. `run_benchmark.py`):

```python title="run_benchmark.py"
# run_benchmark.py
from open_eeg_bench import benchmark

results = benchmark(
    model_cls="my_package.MyModel",
    checkpoint_url="https://my-weights.pth",
    infra={
        "folder": "./results",          # cache directory (stores results + job metadata)
        "cluster": "slurm",             # submit each experiment as a SLURM job
        "gpus_per_node": 1,             # SLURM resource requests
        "mem_gb": 32,
        "timeout_min": 120,
        "cpus_per_task": 4,
        "slurm_partition": "gpu",       # adapt to your cluster
        "slurm_account": "my_account",  # adapt to your cluster
    },
    max_workers=16,  # max SLURM jobs running simultaneously (default: 256)
)
print(results)
```

## 2. Launch from the login node

```bash
python run_benchmark.py
```

That's it. The script itself runs on the login node (no GPU needed) and submits a SLURM **job array** under the hood. Each experiment (dataset x finetuning x head x seed) becomes one job in the array.

## 3. Collect results

When you run the same script again later, already-completed experiments are read from the cache (`./results/`) and only missing or failed experiments are resubmitted. This makes the launcher script **idempotent**: you can rerun it as many times as you want.

## How it works under the hood

- `benchmark()` calls `run_many()`, which uses `exca.TaskInfra.job_array()` to submit all experiments as a single SLURM job array.
- Each experiment is identified by a deterministic **UID** derived from its full configuration (backbone, dataset, seed, finetuning, etc.). Results are cached to `folder/<uid>/`.
- The `max_workers` parameter maps to SLURM's `--array=0-N%max_workers` syntax, controlling how many jobs from the array can run at the same time.
- Logs are written to `<folder>/logs/<username>/<job_id>/` by default.

## Available `infra` options

The `infra` dict accepts the following keys:

| Key | Default | Description |
|-----|---------|-------------|
| `folder` | `None` | **Required for cluster use.** Cache directory for results and job metadata. |
| `cluster` | `None` | `None` (in-process), `"local"` (parallel subprocesses), `"slurm"`, or `"auto"` (SLURM if available, else local). |
| `mode` | `"cached"` | `"cached"` (reuse results), `"retry"` (rerun failures), `"force"` (recompute everything), `"read-only"` (never compute). |
| `gpus_per_node` | `None` | Number of GPUs per job. |
| `cpus_per_task` | `None` | Number of CPUs per job. |
| `mem_gb` | `None` | RAM in GB. |
| `timeout_min` | `None` | Job timeout in minutes. |
| `slurm_partition` | `None` | SLURM partition. |
| `slurm_account` | `None` | SLURM account. |
| `slurm_qos` | `None` | SLURM QoS. |
| `slurm_constraint` | `None` | SLURM node constraint (e.g. `"a100"`). |
| `slurm_additional_parameters` | `None` | Dict of extra SLURM parameters not listed above. |
| `conda_env` | `None` | Name or path of a conda environment to use in the job. |

## Execution modes summary

| `cluster` value | Where it runs | Parallelism | Blocking? |
|-----------------|---------------|-------------|-----------|
| `None` (default) | Current process | Sequential | Yes — returns when all experiments are done. |
| `"local"` | One subprocess per experiment | All at once | No — returns immediately, results collected after. |
| `"slurm"` | SLURM job array | Up to `max_workers` at a time | No — returns immediately, results collected after. |
