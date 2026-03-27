# 🧠 Open EEG Bench

<!-- TODO: uncomment when published
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-47%20passed-brightgreen.svg)]()
-->

**Benchmark any EEG foundation model with one config.**

Open EEG Bench is a plug-and-play framework for evaluating EEG foundation models on downstream tasks using parameter-efficient fine-tuning (PEFT). Bring your own backbone — the framework handles the rest: datasets, splitting, adapter methods, training, and evaluation.

---

## ✨ Features

- 🔌 **Bring your own model** — Wrap any PyTorch EEG model as a backbone and benchmark it instantly
- 🧩 **Combinatorial design** — Mix and match **Backbone x Head x Finetuning Strategy x Dataset** freely
- 📦 **Zero preprocessing** — Datasets are pre-windowed and hosted on [HuggingFace Hub](https://huggingface.co/braindecode)
- ⚡ **7 PEFT methods** — LoRA, IA3, AdaLoRA, DoRA, OFT, full fine-tuning, and frozen linear probing
- 🎯 **One config = one run** — A single `Experiment` object fully describes and executes a training run
- 🧪 **Validated configs** — All configurations are type-checked by [Pydantic](https://docs.pydantic.dev/) at construction time
- 🏗️ **Built on solid foundations** — [braindecode](https://braindecode.org/) + [skorch](https://skorch.readthedocs.io/) + [HuggingFace PEFT](https://huggingface.co/docs/peft)

---

## 🚀 Quick Start

### Installation

```bash
# Requires uv (https://docs.astral.sh/uv/)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Benchmark your model

The `benchmark()` function is the simplest way to evaluate your model. Just point it to your model class and pretrained weights:

```python
from open_eeg_bench import benchmark

results = benchmark(
    model_cls="my_package.MyModel",
    checkpoint_url="https://my-weights.pth",
    peft_target_modules=["encoder.linear1", "encoder.linear2"],
)
print(results)
```

This runs linear probing on **all 12 datasets** and returns a DataFrame with one row per result.

You can pick specific datasets and finetuning strategies:

```python
results = benchmark(
    model_cls="my_package.MyModel",
    checkpoint_url="https://my-weights.pth",
    peft_target_modules=["encoder.linear1", "encoder.linear2"],
    datasets=["arithmetic_zyma2019", "bcic2a", "physionet"],
    finetuning=["frozen", "lora"],
)
```

See the [Benchmarking Your Own Model](#-benchmarking-your-own-model) section for full details on model requirements.

### Advanced: fine-grained control with `Experiment`

For full control over every parameter, use the `Experiment` class directly:

```python
from open_eeg_bench.experiment import Experiment
from open_eeg_bench.default_configs.backbones import biot
from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019
from open_eeg_bench.head import LinearHead
from open_eeg_bench.finetuning import Frozen, LoRA
from open_eeg_bench.training import Training

experiment = Experiment(
    backbone=biot(),
    head=LinearHead(),
    finetuning=Frozen(),
    dataset=arithmetic_zyma2019(),
    training=Training(max_epochs=30, device="cpu"),
)
results = experiment.run()
print(f"Test accuracy: {results['test_balanced_accuracy']:.2%}")
```

---

## 📐 How It Works

Every experiment is defined by **4 independent axes** that you can freely combine:

| Axis | What it controls | Options |
|------|-----------------|---------|
| **Backbone** | The pretrained EEG encoder | BIOT, LaBraM, BENDR, CBraMod, SignalJEPA, REVE, *or your own model* |
| **Head** | The classification/regression layer | `LinearHead`, `MLPHead`, `OriginalHead` |
| **Finetuning** | What gets trained and how | `Frozen`, `FullFinetune`, `LoRA`, `IA3`, `AdaLoRA`, `DoRA`, `OFT` |
| **Dataset** | The downstream evaluation task | Any pre-windowed dataset on HuggingFace Hub |

The `Experiment` class composes them and runs the full pipeline:

```
1. Load windowed dataset from HuggingFace Hub
2. Apply backbone-specific normalization
3. Split into train / val / test
4. Build backbone model + load pretrained weights
5. Replace classification head
6. Apply fine-tuning strategy (PEFT adapters or freeze)
7. Train with braindecode's EEGClassifier (skorch)
8. Evaluate on test set
```

---

## 🔌 Benchmarking Your Own Model

This is the primary use case of Open EEG Bench. You can benchmark **any PyTorch EEG model** with a single function call.

### Model requirements

Your model only needs to:
1. Accept input of shape `(batch, n_chans, n_times)`
2. Return output of shape `(batch, n_outputs)`
3. Have a named module for the classification head (e.g. `self.final_layer`)

### Minimal: `benchmark()`

```python
from open_eeg_bench import benchmark

results = benchmark(
    model_cls="my_package.MyModel",
    model_kwargs=dict(hidden_dim=128),
    checkpoint_url="https://my-model-checkpoint-url.pth",
    peft_target_modules=["encoder.linear1", "encoder.linear2"],
    head_module_name="final_layer",
)
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_cls` | Yes | Dotted import path to your model class |
| `hub_repo` | One of three | HuggingFace Hub repo ID for weights |
| `checkpoint_url` | One of three | URL to pretrained weights |
| `checkpoint_path` | One of three | Local path to pretrained weights |
| `peft_target_modules` | For PEFT | Module names to adapt (e.g. `["to_q", "to_k", "to_v"]`) |
| `model_kwargs` | No | Extra kwargs for the model constructor |
| `head_module_name` | No | Name of the head module (default: `"final_layer"`) |
| `normalization` | No | Post-window normalization |
| `datasets` | No | Dataset names to evaluate on (default: all 12) |
| `finetuning` | No | Strategy names (default: `["frozen"]`) |
| `device` | No | `"cpu"`, `"cuda"`, etc. (default: auto-detect) |

### Recommended: full braindecode compatibility

For full integration with the braindecode ecosystem, your model should inherit from `EEGModuleMixin`:

```python
from braindecode.models.base import EEGModuleMixin
import torch.nn as nn

class MyModel(EEGModuleMixin, nn.Module):
    def __init__(self, n_outputs=None, n_chans=None, n_times=None,
                 chs_info=None, sfreq=None, input_window_seconds=None,
                 hidden_dim=128):
        super().__init__(
            n_outputs=n_outputs, n_chans=n_chans, n_times=n_times,
            chs_info=chs_info, sfreq=sfreq,
            input_window_seconds=input_window_seconds,
        )
        # Your architecture here
        self.encoder = nn.Sequential(
            nn.Conv1d(self.n_chans, hidden_dim, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.final_layer = nn.Linear(hidden_dim, self.n_outputs)

    def forward(self, x):
        # x: (batch, n_chans, n_times)
        features = self.encoder(x)
        return self.final_layer(features)  # (batch, n_outputs)
```

Key rules:
- ✅ Inherit from `EEGModuleMixin` **first**, then `nn.Module`
- ✅ Pass the 6 standard args to `super().__init__()`
- ✅ Name your classification layer `self.final_layer`
- ✅ Output shape: `(batch, n_outputs)` — **no softmax**
- ✅ Input shape: `(batch, n_chans, n_times)`

See the [braindecode contributing guide](https://github.com/braindecode/braindecode/blob/master/CONTRIBUTING.md#adding-a-model-to-braindecode) for the full specification.

---

## 📊 Available Configurations

### 🧠 Built-in Backbones

| Backbone | Reference | Pretrained Source | Normalization |
|----------|-----------|-------------------|---------------|
| **BIOT** | [Yang et al. 2023](https://arxiv.org/abs/2305.10351) | `braindecode/biot-pretrained-prest-16chs` | Percentile scale (q=95) |
| **LaBraM** | [Jiang et al. 2024](https://arxiv.org/abs/2405.18765) | `braindecode/labram-pretrained` | Divide by 100 |
| **BENDR** | [Kostas et al. 2021](https://doi.org/10.3389/fnhum.2021.653659) | `braindecode/braindecode-bendr` | Min-max scale [-1, 1] |
| **CBraMod** | [Wang et al. 2025](https://arxiv.org/abs/2412.07236) | `braindecode/cbramod-pretrained` | Divide by 100 |
| **SignalJEPA** | [Guetschel et al. 2024](https://arxiv.org/abs/2403.11772) | HuggingFace | None |
| **REVE** | [Music et al. 2025](https://arxiv.org/abs/2510.21585) | `brain-bzh/reve-base` | Z-score (clip σ=15) |

### ⚡ Fine-tuning Strategies

| Strategy | Description | Trainable params |
|----------|-------------|-----------------|
| `Frozen()` | Freeze encoder, train only the head | ~0.01% |
| `LoRA(r, alpha)` | Low-Rank Adaptation | ~1-5% |
| `IA3()` | Inhibiting and Amplifying Inner Activations | ~0.1% |
| `AdaLoRA(r, target_r)` | Adaptive rank allocation | ~1-5% |
| `DoRA(r, alpha)` | Weight-Decomposed LoRA | ~1-5% |
| `OFT(block_size)` | Orthogonal Fine-Tuning | ~1-10% |
| `FullFinetune()` | Train all parameters | 100% |

### 📦 Available Datasets

All 12 datasets are pre-windowed and hosted on [HuggingFace Hub](https://huggingface.co/braindecode):

| Dataset | HF ID | Classes | Task |
|---------|-------|---------|------|
| Arithmetic (Zyma 2019) | `braindecode/arithmetic_zyma2019` | 2 | Mental arithmetic vs. rest |
| BCI Competition IV 2a | `braindecode/bcic2a` | 4 | Motor imagery |
| BCI Competition 2020-3 | `braindecode/bcic2020-3` | 5 | Imagined speech |
| PhysioNet MI | `braindecode/physionet` | 4 | Motor imagery |
| CHB-MIT | `braindecode/chbmit` | 2 | Seizure detection |
| FACED | `braindecode/faced` | 9 | Emotion recognition |
| ISRUC-Sleep | `braindecode/isruc-sleep` | 5 | Sleep staging |
| MDD (Mumtaz 2016) | `braindecode/mdd_mumtaz2016` | 2 | Depression detection |
| SEED-V | `braindecode/seed-v` | 5 | Emotion recognition |
| SEED-VIG | `braindecode/seed-vig` | regression | Vigilance estimation |
| TUAB | `braindecode/tuab` | 2 | Abnormal EEG detection |
| TUEV | `braindecode/tuev` | 6 | EEG event classification |

---

## 🏗️ Architecture Overview

```
Experiment
├── seed
├── backbone ─────── BIOT / LaBraM / BENDR / ... / YourModel
│   └── normalization ── PercentileScale / DivideByConstant / ...
├── head ─────────── LinearHead / MLPHead / OriginalHead
├── finetuning ───── LoRA / IA3 / Frozen / FullFinetune / ...
├── dataset ──────── HF dataset ID + Splitter
│   └── splitter ──── RandomSplitter / CrossSubjectSplitter / PredefinedSplitter
└── training ─────── epochs, lr, early stopping, wandb, ...
```

Each component is a **Pydantic config class** that carries both its parameters and its logic. No YAML files, no Hydra, no indirection — what you see is what you get.

---

## 🧪 Running Tests

```bash
pytest                                # all tests (47 tests)
pytest tests/test_default_configs.py  # config validation
pytest tests/test_backbones.py        # model build & forward
pytest tests/test_normalization.py    # normalization correctness
pytest -k "test_name"                 # single test
```

---

## 🤝 Contributing

Contributions are welcome! Here are some ways to help:

- **Add a new backbone** — Follow the [guide above](#-benchmarking-your-own-model) and submit a PR with your backbone config + default config factory function
- **Add a new dataset** — Push a pre-windowed `BaseConcatDataset` to HuggingFace Hub and add a factory function in `default_configs/datasets.py`
- **Add a new fine-tuning strategy** — Implement a new class in `finetuning.py` inheriting from `BaseModel` with `apply()` and `get_callbacks()` methods
- **Report bugs or suggest features** — Open an issue on GitHub

### Development setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest  # make sure everything passes
```

---

## 📄 License

MIT

---

## 📚 References

If you use Open EEG Bench in your research, please cite:

<!-- TODO: Add citation when paper is published -->

### PEFT Methods
- **LoRA**: [Hu et al. (2021)](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation of Large Language Models
- **IA3**: [Liu et al. (2022)](https://arxiv.org/abs/2205.05638) — Few-Shot Parameter-Efficient Fine-Tuning
- **AdaLoRA**: [Zhang et al. (2023)](https://arxiv.org/abs/2303.10512) — Adaptive Budget Allocation for PEFT
- **DoRA**: [Liu et al. (2024)](https://arxiv.org/abs/2402.09353) — Weight-Decomposed Low-Rank Adaptation
- **OFT**: [Qiu et al. (2023)](https://arxiv.org/abs/2306.07280) — Orthogonal Fine-Tuning

### Key Libraries
- [braindecode](https://braindecode.org/) — Deep learning for EEG
- [skorch](https://skorch.readthedocs.io/) — scikit-learn compatible neural network training
- [HuggingFace PEFT](https://huggingface.co/docs/peft) — Parameter-Efficient Fine-Tuning
- [Pydantic](https://docs.pydantic.dev/) — Data validation using Python type annotations
