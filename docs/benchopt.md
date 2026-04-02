# Running with Benchopt

[Benchopt](https://benchopt.github.io/) is a framework for reproducible and collaborative benchmarks in optimization and machine learning. It provides a standardized way to define datasets, solvers, and objectives, and handles running all combinations, collecting results, and generating comparison plots.

Open EEG Bench provides a **Benchopt interface** as an alternative to the Python API (`benchmark()` function) and `exca`-based workflow. Both approaches run the same underlying code — only the way you launch jobs, iterate over parameter combinations, and collect results differs.

| | Python API / exca | Benchopt |
|---|---|---|
| Launch | `benchmark()` or `run_many()` | `benchopt run` CLI |
| Parameter grid | Manual loops or `make_all_experiments()` | Automatic from `parameters` dicts |
| Caching | exca `TaskInfra` | Benchopt's built-in caching |
| SLURM | `infra={"cluster": "slurm"}` | `benchopt run --parallel-config my_config_parallel.yaml` (see [doc](https://benchopt.github.io/stable/user_guide/distributed_run.html)) |
| Results | `pd.DataFrame` | `.parquet` files + built-in plots |

## Installation

Install Open EEG Bench with the `benchopt` optional dependency:

```bash
pip install open-eeg-bench[benchopt]
```

Or, if you already have the package installed:

```bash
pip install benchopt
```

## Quick start

Run a single backbone + finetuning combination on one dataset:

```bash
benchopt run ./benchopt_wrappers \
  -s "FineTune[backbone_name=biot,finetuning_name=frozen,head_name=linear_head]" \
  -d "OpenEEG[dataset_name=arithmetic_zyma2019]" \
  --n-repetitions 3
```

Run the full benchmark (all backbones x finetuning strategies x heads x datasets):

```bash
benchopt run ./benchopt_wrappers --n-repetitions 3
```

Results are saved as `.parquet` files in `benchopt_wrappers/outputs/`.

## Selecting parameters

The benchmark exposes three parameter axes on the solver side and one on the dataset side. Use `-s` and `-d` flags to filter:

**Solver parameters** (`-s`):

| Parameter | Values |
|-----------|--------|
| `backbone_name` | `biot`, `labram`, `bendr`, `cbramod`, `signal_jepa`, `reve` |
| `finetuning_name` | `frozen`, `lora`, `ia3`, `adalora`, `dora`, `oft`, `full_finetune`, `two_stages` |
| `head_name` | `linear_head`, `mlp_head`, `original_head` |

**Dataset parameters** (`-d`):

| Parameter | Values |
|-----------|--------|
| `dataset_name` | `arithmetic_zyma2019`, `bcic2a`, `bcic2020_3`, `physionet`, `chbmit`, `faced`, `isruc_sleep`, `mdd_mumtaz2016`, `seed_v`, `seed_vig`, `tuab`, `tuev` |

Examples:

```bash
# Two backbones, frozen only, on two datasets
benchopt run ./benchopt_wrappers \
  -s "FineTune[backbone_name=biot,finetuning_name=frozen,head_name=linear_head]" \
  -s "FineTune[backbone_name=labram,finetuning_name=frozen,head_name=linear_head]" \
  -d "OpenEEG[dataset_name=arithmetic_zyma2019]" \
  -d "OpenEEG[dataset_name=bcic2a]"

# LoRA vs frozen on all datasets
benchopt run ./benchopt_wrappers \
  -s "FineTune[backbone_name=biot,finetuning_name=frozen,head_name=linear_head]" \
  -s "FineTune[backbone_name=biot,finetuning_name=lora,head_name=linear_head]"
```

## How it maps to Open EEG Bench

The Benchopt interface is a thin wrapper around existing classes:

- **`Objective`** (`objective.py`): receives the dataset config and evaluates the result dict returned by `Experiment.run()` (balanced accuracy for classification, R2 for regression).
- **`Dataset`** (`datasets/open_eeg.py`): parameterized by `dataset_name`, instantiates the corresponding `Dataset` config from `open_eeg_bench.default_configs.datasets`.
- **`Solver`** (`solvers/finetune.py`): parameterized by `backbone_name`, `finetuning_name`, and `head_name`. Constructs an `Experiment` from the default configs and calls `experiment.run()`.

No training logic is duplicated — the solver delegates entirely to `Experiment.run()`.
