# 🧠 Open EEG Bench

**Benchmark any EEG foundation model with one function call.**

## ✨ Why Open EEG Bench?

- 🎯 **One function, all results** — Call `benchmark()` with your model and get a full evaluation across 12 datasets
- 📦 **Zero preprocessing** — All datasets are pre-windowed and hosted on [HuggingFace Hub](https://huggingface.co/braindecode), ready to use
- ⚡ **7 fine-tuning strategies** — Frozen linear probing, LoRA, IA3, AdaLoRA, DoRA, OFT, and full fine-tuning
- 🔌 **Bring your own model** — Any PyTorch model that takes EEG input and returns features works out of the box
- 🔒 **Reproducible by design** — A single config object fully describes a run. No YAML files, no hidden state

## 🚀 Installation

```bash
pip install open-eeg-bench
```

## 🏁 Benchmark your model

```python
from open_eeg_bench import benchmark

results = benchmark(
    model_cls="my_package.MyModel",  # import path to your model class
    checkpoint_url="https://my-weights.pth",
)

print(results)  # pd.DataFrame
```

This runs linear probing on **all 12 datasets** and returns a DataFrame with one row per result.

You can pick specific datasets, fine-tuning strategies, classification heads and number of initialization seeds:

```python
results = benchmark(
    model_cls="my_package.MyModel",
    checkpoint_url="https://my-weights.pth",
    datasets=["arithmetic_zyma2019", "bcic2a", "physionet"],
    finetuning_strategies=["frozen", "lora"],
    peft_target_modules=[  # necessary for LoRA, IA3, AdaLoRA, OFT, and DoRA
        "encoder.linear1", 
        "encoder.linear2"
    ],
    heads=["linear_head", "mlp_head"],
    n_seeds=5,
)
```

Need to run on a SLURM cluster? No `sbatch` scripts needed — see [Running on a cluster](docs/cluster.md).

## 📐 Model requirements

Your model only needs to:

1. Accept input of shape `(batch, n_chans, n_times)`
2. Return output of shape `(batch, n_outputs)`
3. Have a named module for the classification head (default: `self.final_layer`)

## ⚙️ `benchmark()` parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_cls` | Yes | Dotted import path to your model class |
| `hub_repo` | One of three | HuggingFace Hub repo ID for weights |
| `checkpoint_url` | One of three | URL to pretrained weights |
| `checkpoint_path` | One of three | Local path to pretrained weights |
| `peft_target_modules` | For PEFT | Module names to adapt (e.g. `["to_q", "to_k", "to_v"]`) |
| `peft_ff_modules` | For IA3 | Feedforward module names for IA3 adapter |
| `model_kwargs` | No | Extra kwargs for the model constructor |
| `head_module_name` | No | Name of the head module (default: `"final_layer"`) |
| `normalization` | No | Post-window normalization |
| `datasets` | No | Dataset names to evaluate on (default: all 12) |
| `heads` | No | Head names: `"linear_head"`, `"mlp_head"`, `"original_head"` (default: `["linear_head"]`) |
| `finetuning_strategies` | No | Strategy names (default: `["frozen"]`) |
| `n_seeds` | No | Number of random seeds (default: `3`) |
| `device` | No | `"cpu"`, `"cuda"`, etc. (default: `"cpu"`) |
| `infra` | No | Infrastructure config for caching and cluster submission (see [cluster docs](docs/cluster.md)) |
| `max_workers` | No | Max simultaneous SLURM jobs (default: `256`) |

## 📊 Available datasets

All 12 datasets are pre-windowed and hosted on [HuggingFace Hub](https://huggingface.co/braindecode):

| Dataset | HF ID | Classes | Window size | Task |
|---------|-------|---------|------------|------|
| Arithmetic (Zyma 2019) | [`braindecode/arithmetic_zyma2019`](https://huggingface.co/datasets/braindecode/arithmetic_zyma2019) | 2 | 5 s | Mental arithmetic vs. rest |
| BCI Competition IV 2a | [`braindecode/bcic2a`](https://huggingface.co/datasets/braindecode/bcic2a) | 4 | 4 s | Motor imagery |
| BCI Competition 2020-3 | [`braindecode/bcic2020-3`](https://huggingface.co/datasets/braindecode/bcic2020-3) | 5 | 3 s | Imagined speech |
| PhysioNet MI | [`braindecode/physionet`](https://huggingface.co/datasets/braindecode/physionet) | 4 | 3 s | Motor imagery |
| CHB-MIT | [`braindecode/chbmit`](https://huggingface.co/datasets/braindecode/chbmit) | 2 | 10 s | Seizure detection |
| FACED | [`braindecode/faced`](https://huggingface.co/datasets/braindecode/faced) | 9 | 10 s | Emotion recognition |
| ISRUC-Sleep | [`braindecode/isruc-sleep`](https://huggingface.co/datasets/braindecode/isruc-sleep) | 5 | 30 s | Sleep staging |
| MDD (Mumtaz 2016) | [`braindecode/mdd_mumtaz2016`](https://huggingface.co/datasets/braindecode/mdd_mumtaz2016) | 2 | 5 s | Depression detection |
| SEED-V | [`braindecode/seed-v`](https://huggingface.co/datasets/braindecode/seed-v) | 5 | 1 s | Emotion recognition |
| SEED-VIG | [`braindecode/seed-vig`](https://huggingface.co/datasets/braindecode/seed-vig) | regression | 8 s | Vigilance estimation |
| TUAB | [`braindecode/tuab`](https://huggingface.co/datasets/braindecode/tuab) | 2 | 10 s | Abnormal EEG detection |
| TUEV | [`braindecode/tuev`](https://huggingface.co/datasets/braindecode/tuev) | 6 | 5 s | EEG event classification |

**Preprocessing:**  Window lengths are dataset-dependant (see table above). All datasets are high-pass filtered at 0.1 Hz, except for tasks with short trial windows (2 s or less), where we use 0.5 Hz. All datasets are resampled to 100 Hz. Model-specific normalization (e.g. z-scoring) can be applied via the `normalization` parameter.

## 🧩 Available fine-tuning strategies

| Strategy | Description | Trainable params |
|----------|-------------|-----------------|
| `Frozen()` | Freeze encoder, train only the head | ~0.01% |
| `LoRA(r, alpha)` | Low-Rank Adaptation | ~1-5% |
| `IA3()` | Inhibiting and Amplifying Inner Activations | ~0.1% |
| `AdaLoRA(r, target_r)` | Adaptive rank allocation | ~1-5% |
| `DoRA(r, alpha)` | Weight-Decomposed LoRA | ~1-5% |
| `OFT(block_size)` | Orthogonal Fine-Tuning | ~1-10% |
| `FullFinetune()` | Train all parameters | 100% |
| `TwoStages()` | Frozen head for 10 epochs, then unfreeze and train all | 100% |

## 📄 License

BSD-3-Clause (see [LICENSE.txt](LICENSE.txt))

## 📚 Further reading

- [Running on a cluster](docs/cluster.md) — SLURM submission, `infra` options, caching, execution modes
- [Running with Benchopt](docs/benchopt.md) — Alternative CLI-based workflow using the [Benchopt](https://benchopt.github.io/) framework
- [Advanced usage](docs/advanced.md) — `Experiment` class, architecture overview, braindecode integration
- [Built-in backbones](docs/backbones.md) — Pretrained models shipped with the benchmark
- [Contributing](docs/contributing.md) — How to add backbones, datasets, or fine-tuning strategies
- [References](docs/references.md) — Citations and key libraries
