from open_eeg_bench.default_configs.backbones import ALL_BACKBONES
from open_eeg_bench.default_configs.datasets import ALL_DATASETS
from open_eeg_bench.finetuning import (
    AdaLoRA,
    DoRA,
    Frozen,
    FullFinetune,
    IA3,
    LoRA,
    OFT,
    TwoStages,
)

ALL_FINETUNING = {
    "lora": LoRA,
    "ia3": IA3,
    "adalora": AdaLoRA,
    "dora": DoRA,
    "oft": OFT,
    "full_finetune": FullFinetune,
    "frozen": Frozen,
    'two_stages': TwoStages,
}

from open_eeg_bench.head import LinearHead, MLPHead, OriginalHead

ALL_HEADS = {
    "linear_head": LinearHead,
    "mlp_head": MLPHead,
    "original_head": OriginalHead,
}

from open_eeg_bench.default_configs.experiments import make_all_experiments
