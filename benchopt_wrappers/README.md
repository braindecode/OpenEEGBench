# Benchopt interface for Open EEG Bench

This directory contains the [Benchopt](https://benchopt.github.io/) wrapper for Open EEG Bench.

See [docs/benchopt.md](../docs/benchopt.md) for installation instructions and usage.

Quick start:

```bash
pip install open-eeg-bench[benchopt]

benchopt run ./benchmark_open_eeg \
  -s "FineTune[backbone_name=biot,finetuning_name=frozen,head_name=linear_head]" \
  -d "OpenEEG[dataset_name=arithmetic_zyma2019]" \
  --n-repetitions 3
```
