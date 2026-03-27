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
# Requires uv (https://docs.astral.sh/uv/)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## 🏁 Benchmark your model

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

You can pick specific datasets and fine-tuning strategies:

```python
results = benchmark(
    model_cls="my_package.MyModel",
    checkpoint_url="https://my-weights.pth",
    peft_target_modules=["encoder.linear1", "encoder.linear2"],
    datasets=["arithmetic_zyma2019", "bcic2a", "physionet"],
    finetuning=["frozen", "lora"],
)
```

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
| `model_kwargs` | No | Extra kwargs for the model constructor |
| `head_module_name` | No | Name of the head module (default: `"final_layer"`) |
| `normalization` | No | Post-window normalization |
| `datasets` | No | Dataset names to evaluate on (default: all 12) |
| `finetuning` | No | Strategy names (default: `["frozen"]`) |
| `device` | No | `"cpu"`, `"cuda"`, etc. (default: auto-detect) |

## 📊 Available datasets

All 12 datasets are pre-windowed and hosted on [HuggingFace Hub](https://huggingface.co/braindecode):

| Dataset | HF ID | Classes | Task |
|---------|-------|---------|------|
| Arithmetic (Zyma 2019) | [`braindecode/arithmetic_zyma2019`](https://huggingface.co/datasets/braindecode/arithmetic_zyma2019) | 2 | Mental arithmetic vs. rest |
| BCI Competition IV 2a | [`braindecode/bcic2a`](https://huggingface.co/datasets/braindecode/bcic2a) | 4 | Motor imagery |
| BCI Competition 2020-3 | [`braindecode/bcic2020-3`](https://huggingface.co/datasets/braindecode/bcic2020-3) | 5 | Imagined speech |
| PhysioNet MI | [`braindecode/physionet`](https://huggingface.co/datasets/braindecode/physionet) | 4 | Motor imagery |
| CHB-MIT | [`braindecode/chbmit`](https://huggingface.co/datasets/braindecode/chbmit) | 2 | Seizure detection |
| FACED | [`braindecode/faced`](https://huggingface.co/datasets/braindecode/faced) | 9 | Emotion recognition |
| ISRUC-Sleep | [`braindecode/isruc-sleep`](https://huggingface.co/datasets/braindecode/isruc-sleep) | 5 | Sleep staging |
| MDD (Mumtaz 2016) | [`braindecode/mdd_mumtaz2016`](https://huggingface.co/datasets/braindecode/mdd_mumtaz2016) | 2 | Depression detection |
| SEED-V | [`braindecode/seed-v`](https://huggingface.co/datasets/braindecode/seed-v) | 5 | Emotion recognition |
| SEED-VIG | [`braindecode/seed-vig`](https://huggingface.co/datasets/braindecode/seed-vig) | regression | Vigilance estimation |
| TUAB | [`braindecode/tuab`](https://huggingface.co/datasets/braindecode/tuab) | 2 | Abnormal EEG detection |
| TUEV | [`braindecode/tuev`](https://huggingface.co/datasets/braindecode/tuev) | 6 | EEG event classification |

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

## 📄 License

MIT

## 📚 Further reading

- [Advanced usage](docs/advanced.md) — `Experiment` class, architecture overview, braindecode integration
- [Built-in backbones](docs/backbones.md) — Pretrained models shipped with the benchmark
- [Contributing](docs/contributing.md) — How to add backbones, datasets, or fine-tuning strategies
- [References](docs/references.md) — Citations and key libraries
