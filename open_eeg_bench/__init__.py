"""Open EEG Bench — Benchmarking parameter-efficient fine-tuning of EEG foundation models."""

from open_eeg_bench.main import benchmark
from open_eeg_bench import default_configs, experiment, finetuning, head, training

__all__ = [
    "benchmark",
    "default_configs",
    "experiment",
    "finetuning",
    "head",
    "training",
]
