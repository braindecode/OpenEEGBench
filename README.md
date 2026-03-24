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
- 🧩 **Combinatorial design** — Mix and match **Backbone × Head × Finetuning Strategy × Dataset** freely
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

### Run your first experiment

```python
from open_eeg_bench.experiment import Experiment
from open_eeg_bench.default_configs.backbones import biot
from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019
from open_eeg_bench.head import LinearHead
from open_eeg_bench.finetuning import Frozen, LoRA
from open_eeg_bench.training import Training

# Linear probing: freeze the backbone, train only the head
experiment = Experiment(
    backbone=biot(),
    head=LinearHead(),
    finetuning=Frozen(),
    dataset=arithmetic_zyma2019(),
    training=Training(max_epochs=30, device="cpu"),
)
results = experiment.run()
print(f"Test accuracy: {results['test_balanced_accuracy']:.2%}")

# Or try LoRA fine-tuning instead — just swap one line:
experiment = Experiment(
    backbone=biot(),
    head=LinearHead(),
    finetuning=LoRA(r=16, alpha=32),
    dataset=arithmetic_zyma2019(),
    training=Training(max_epochs=30, device="cpu"),
)
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

This is the primary use case of Open EEG Bench. You can wrap **any PyTorch EEG model** and benchmark it against the built-in backbones.

### Minimal: just make it work

Your model only needs to:
1. Accept input of shape `(batch, n_chans, n_times)`
2. Return output of shape `(batch, n_outputs)`
3. Have a named module for the classification head (e.g. `self.final_layer`)

```python
import torch.nn as nn
from pydantic import Literal
from open_eeg_bench.backbone import _BackboneBase

class MyModelBackbone(_BackboneBase):
    kind: Literal["my_model"] = "my_model"

    # Your architecture hyperparameters
    hidden_dim: int = 128

    # Tell the framework about your model
    peft_target_modules: list[str] = ["encoder.linear1", "encoder.linear2"]
    peft_ff_modules: list[str] = ["encoder.linear2"]
    head_module_name: str = "final_layer"
    hub_repo: str | None = None  # or your HF repo with pretrained weights

    def _model_class(self):
        return MyModel  # your nn.Module class

    def _model_kwargs(self):
        return dict(hidden_dim=self.hidden_dim)
```

Then use it like any other backbone:

```python
from open_eeg_bench.experiment import Experiment
from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019

results = Experiment(
    backbone=MyModelBackbone(),
    dataset=arithmetic_zyma2019(),
).run()
```

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

All datasets are pre-windowed and hosted on [HuggingFace Hub](https://huggingface.co/braindecode):

| Dataset | HF ID | Classes | Task |
|---------|-------|---------|------|
| Arithmetic (Zyma 2019) | `braindecode/arithmetic_zyma2019` | 2 | Mental arithmetic vs. rest |
| BCI Competition IV 2a | `braindecode/bcic2a` | 4 | Motor imagery |
| PhysioNet MI | `braindecode/physionet` | 4 | Motor imagery |

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
