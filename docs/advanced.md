# Advanced Usage

## Fine-grained control with `Experiment`

The `benchmark()` function covers most use cases, but if you need full control over every parameter, use the `Experiment` class directly:

```python
import open_eeg_bench as oeb

experiment = oeb.experiment.Experiment(
    backbone=oeb.default_configs.backbones.biot(),
    head=oeb.head.LinearHead(),
    finetuning=oeb.finetuning.Frozen(),
    dataset=oeb.default_configs.datasets.arithmetic_zyma2019(),
    training=oeb.training.Training(max_epochs=30, device="cpu"),
)
results = experiment.run()
print(f"Test accuracy: {results['test_balanced_accuracy']:.2%}")
```

---

## How It Works

Every experiment is defined by **4 independent axes** that you can freely combine:

| Axis | What it controls | Options |
|------|-----------------|---------|
| **Backbone** | The pretrained EEG encoder | BIOT, LaBraM, BENDR, CBraMod, SignalJEPA, REVE, *or your own model* |
| **Head** | The classification/regression layer | `LinearHead`, `MLPHead`, `OriginalHead` |
| **Finetuning** | What gets trained and how | `Frozen`, `FullFinetune`, `LoRA`, `IA3`, `AdaLoRA`, `DoRA`, `OFT`, `TwoStages` |
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

## Architecture Overview

```
Experiment
├── seed
├── backbone ─────── BIOT / LaBraM / BENDR / ... / YourModel
│   └── normalization ── PercentileScale / DivideByConstant / ...
├── head ─────────── LinearHead / MLPHead / OriginalHead
├── finetuning ───── LoRA / IA3 / Frozen / FullFinetune / ...
├── dataset ──────── HF dataset ID + Splitter
│   └── splitter ──── PredefinedSplitter
└── training ─────── epochs, lr, early stopping, wandb, ...
```

Each component is a **Pydantic config class** that carries both its parameters and its logic. No YAML files, no Hydra, no indirection — what you see is what you get.

---

## Full braindecode compatibility

For full integration with the braindecode ecosystem, your model can inherit from `EEGModuleMixin`:

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
- Inherit from `EEGModuleMixin` **first**, then `nn.Module`
- Pass the 6 standard args to `super().__init__()`
- Name your classification layer `self.final_layer`
- Output shape: `(batch, n_outputs)` — **no softmax**
- Input shape: `(batch, n_chans, n_times)`

See the [braindecode contributing guide](https://github.com/braindecode/braindecode/blob/master/CONTRIBUTING.md#adding-a-model-to-braindecode) for the full specification.

---

## Running on a cluster

Both `benchmark()` and `Experiment` support SLURM submission via the `infra` parameter — no `sbatch` scripts needed. See [Running on a cluster](cluster.md) for the full guide on `infra` options, caching, and execution modes.
