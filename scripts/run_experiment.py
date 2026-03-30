#!/usr/bin/env python
"""Run the BIOT + arithmetic_zyma2019 + linear probing experiment."""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from open_eeg_bench.experiment import Experiment
from open_eeg_bench.default_configs.backbones import biot
from open_eeg_bench.default_configs.datasets import arithmetic_zyma2019
from open_eeg_bench.head import LinearHead
from open_eeg_bench.finetuning import Frozen
from open_eeg_bench.training import Training, EarlyStopping

experiment = Experiment(
    seed=42,
    backbone=biot(),
    head=LinearHead(),
    finetuning=Frozen(),
    dataset=arithmetic_zyma2019(),
    training=Training(
        max_epochs=30,
        lr=1e-3,
        device="cpu",
        early_stopping=EarlyStopping(patience=10, monitor="valid_loss"),
    ),
)

results = experiment.run()
print(f"\nResults: {results}")
if "test_balanced_accuracy" in results:
    acc = results["test_balanced_accuracy"]
    print(f"Test balanced accuracy: {acc:.4f} (chance = 0.50)")
    print("ABOVE CHANCE" if acc > 0.50 else "BELOW CHANCE — needs tuning")
