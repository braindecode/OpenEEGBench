"""Default fine-tuning strategy configurations."""

from open_eeg_bench.finetuning import (
    AdaLoRA,
    DoRA,
    Frozen,
    FullFinetune,
    IA3,
    LoRA,
    OFT,
)


def lora(**overrides) -> LoRA:
    return LoRA(**overrides)


def ia3(**overrides) -> IA3:
    return IA3(**overrides)


def adalora(**overrides) -> AdaLoRA:
    return AdaLoRA(**overrides)


def dora(**overrides) -> DoRA:
    return DoRA(**overrides)


def oft(**overrides) -> OFT:
    return OFT(**overrides)


def full_finetune(**overrides) -> FullFinetune:
    return FullFinetune(**overrides)


def frozen(**overrides) -> Frozen:
    return Frozen(**overrides)


ALL_FINETUNING = [lora, ia3, adalora, dora, oft, full_finetune, frozen]
