"""Benchmark a built-in backbone on multiple datasets with multiple strategies.

This example uses the BIOT backbone factory from default_configs, which
bundles the correct model_kwargs, normalization, and PEFT target modules.
"""

from open_eeg_bench import benchmark
from open_eeg_bench.default_configs.backbones import biot

backbone = biot()

results = benchmark(
    model_cls=backbone.model_cls,
    hub_repo=backbone.hub_repo,
    model_kwargs=backbone.model_kwargs,
    peft_target_modules=backbone.peft_target_modules,
    peft_ff_modules=backbone.peft_ff_modules,
    normalization=backbone.normalization,
    datasets=["arithmetic_zyma2019", "bcic2a"],
    finetuning_strategies=["frozen", "lora"],
    n_seeds=3,
    device="cuda",
)

print(results)
