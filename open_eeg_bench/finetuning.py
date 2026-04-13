"""Fine-tuning strategy configurations.

Each strategy defines what to freeze/adapt and how.  PEFT-based strategies
(LoRA, IA3, …) wrap the model with HuggingFace PEFT.  ``Frozen`` freezes
the encoder and returns a skorch ``Freezer`` callback.  ``FullFinetune``
trains everything.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import torch.nn as nn
    from open_eeg_bench.backbone import _BackboneBase

log = logging.getLogger(__name__)


def _param_stats(model) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }


def _resolve_modules_to_save(model, head_module_name: str) -> list[str] | None:
    """Auto-detect the head module to keep trainable under PEFT."""
    if hasattr(model, head_module_name):
        return [head_module_name]
    for candidate in ("final_layer", "classifier", "head"):
        if hasattr(model, candidate):
            return [candidate]
    return None


def _filter_linear_targets(model, target_modules: list[str]) -> list[str]:
    """Keep only target_modules that resolve to Linear/Conv layers in the model.

    Some adapters (IA3, OFT) only support Linear/Conv, not MultiheadAttention.
    """
    import torch.nn as nn

    supported = (nn.Linear, nn.Conv1d, nn.Conv2d)
    valid = set()
    for name, module in model.named_modules():
        if not isinstance(module, supported):
            continue
        for target in target_modules:
            if name == target or name.endswith("." + target):
                valid.add(target)
    if not valid:
        return target_modules  # fallback: let PEFT raise the original error
    filtered_out = set(target_modules) - valid
    if filtered_out:
        log.info(
            "Filtered out unsupported target modules: %s (keeping: %s)",
            sorted(filtered_out), sorted(valid),
        )
    return sorted(valid)


def _apply_peft(model, peft_config):
    """Apply PEFT config to the model."""
    from peft import get_peft_model

    peft_model = get_peft_model(model, peft_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    return peft_model, trainable, total


# ============================================================================
# Concrete strategies
# ============================================================================


class LoRA(BaseModel):
    """Low-Rank Adaptation."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["lora"] = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    bias: str = "all"

    def apply(self, model, backbone):
        from peft import LoraConfig as PeftLoraConfig

        modules_to_save = _resolve_modules_to_save(model, backbone.head_module_name)
        cfg = PeftLoraConfig(
            r=self.r, lora_alpha=self.alpha, lora_dropout=self.dropout,
            target_modules=backbone.peft_target_modules,
            bias=self.bias, modules_to_save=modules_to_save,
        )
        wrapped, trainable, total = _apply_peft(model, cfg)
        return wrapped, {"method": "lora", "total_params": total,
                         "trainable_params": trainable, "trainable_pct": 100.0 * trainable / total}

    def get_callbacks(self) -> list:
        return []


class IA3(BaseModel):
    """Infused Adapter by Inhibiting and Amplifying Inner Activations."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["ia3"] = "ia3"

    def apply(self, model, backbone):
        from peft import IA3Config as PeftIA3Config

        modules_to_save = _resolve_modules_to_save(model, backbone.head_module_name)
        ff_modules = backbone.peft_ff_modules or None
        target_modules = list(backbone.peft_target_modules)
        if ff_modules:
            target_modules = list(set(target_modules) | set(ff_modules))
        target_modules = _filter_linear_targets(model, target_modules)
        if ff_modules:
            ff_modules = [m for m in ff_modules if m in target_modules]
        cfg = PeftIA3Config(
            target_modules=target_modules, feedforward_modules=ff_modules or None,
            modules_to_save=modules_to_save,
        )
        wrapped, trainable, total = _apply_peft(model, cfg)
        return wrapped, {"method": "ia3", "total_params": total,
                         "trainable_params": trainable, "trainable_pct": 100.0 * trainable / total}

    def get_callbacks(self) -> list:
        return []


class AdaLoRA(BaseModel):
    """Adaptive Low-Rank Adaptation with dynamic rank allocation."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["adalora"] = "adalora"
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_r: int = 4
    init_r: int = 12
    total_step: int | None = None
    tinit: int = 200
    tfinal: int = 1000
    deltaT: int = 10

    def apply(self, model, backbone):
        from peft import AdaLoraConfig as PeftAdaLoraConfig

        target_modules = _filter_linear_targets(model, list(backbone.peft_target_modules))
        cfg = PeftAdaLoraConfig(
            r=self.r, lora_alpha=self.alpha, lora_dropout=self.dropout,
            target_modules=target_modules,
            total_step=self.total_step, target_r=self.target_r, init_r=self.init_r,
            tinit=self.tinit, tfinal=self.tfinal, deltaT=self.deltaT,
        )
        wrapped, trainable, total = _apply_peft(model, cfg)
        return wrapped, {"method": "adalora", "total_params": total,
                         "trainable_params": trainable, "trainable_pct": 100.0 * trainable / total}

    def get_callbacks(self) -> list:
        return []


class DoRA(BaseModel):
    """Weight-Decomposed Low-Rank Adaptation."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["dora"] = "dora"
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    bias: str = "none"

    def apply(self, model, backbone):
        from peft import LoraConfig as PeftLoraConfig

        modules_to_save = _resolve_modules_to_save(model, backbone.head_module_name)
        # DoRA does not support MultiheadAttention; filter to Linear/Conv only
        target_modules = _filter_linear_targets(model, list(backbone.peft_target_modules))
        cfg = PeftLoraConfig(
            r=self.r, lora_alpha=self.alpha, lora_dropout=self.dropout,
            target_modules=target_modules,
            bias=self.bias, modules_to_save=modules_to_save, use_dora=True,
        )
        wrapped, trainable, total = _apply_peft(model, cfg)
        return wrapped, {"method": "dora", "total_params": total,
                         "trainable_params": trainable, "trainable_pct": 100.0 * trainable / total}

    def get_callbacks(self) -> list:
        return []


class OFT(BaseModel):
    """Orthogonal Fine-Tuning."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["oft"] = "oft"
    block_size: int = 8
    module_dropout: float = 0.0
    coft: bool = False
    block_share: bool = False

    def apply(self, model, backbone):
        from peft import OFTConfig as PeftOFTConfig

        modules_to_save = _resolve_modules_to_save(model, backbone.head_module_name)
        target_modules = _filter_linear_targets(model, list(backbone.peft_target_modules))
        cfg = PeftOFTConfig(
            oft_block_size=self.block_size, target_modules=target_modules,
            module_dropout=self.module_dropout, coft=self.coft,
            block_share=self.block_share, modules_to_save=modules_to_save,
        )
        wrapped, trainable, total = _apply_peft(model, cfg)
        return wrapped, {"method": "oft", "total_params": total,
                         "trainable_params": trainable, "trainable_pct": 100.0 * trainable / total}

    def get_callbacks(self) -> list:
        return []


class FullFinetune(BaseModel):
    """Train all parameters (no adapters)."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["full"] = "full"

    def apply(self, model, backbone):
        return model, {**_param_stats(model), "method": "full"}

    def get_callbacks(self) -> list:
        return []


class Frozen(BaseModel):
    """Freeze the encoder, train only the head.

    Returns a model in which all parameters
    whose name does *not* contain the head module name are frozen.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["frozen"] = "frozen"

    def apply(self, model, backbone):
        head_name = backbone.head_module_name
        for name, param in model.named_parameters():
            if head_name not in name:
                param.requires_grad = False
        return model, {**_param_stats(model), "method": "frozen"}

    def get_callbacks(self) -> list:
        return []


class TwoStages(BaseModel):
    """Two-stage training: frozen backbone for N epochs, then unfreeze and train all.

    Returns a model in which all parameters
    whose name does *not* contain the head module name are frozen.
    After N epochs, an Unfreezer callback will unfreeze the whole model for the rest of training.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["two_stages"] = "two_stages"
    n_epochs_frozen: int = 5

    def apply(self, model, backbone):
        head_name = backbone.head_module_name
        for name, param in model.named_parameters():
            if head_name not in name:
                param.requires_grad = False
        return model, {**_param_stats(model), "method": "frozen"}

    def get_callbacks(self) -> list:
        from skorch.callbacks import ParamMapper
        from skorch.utils import unfreeze_parameter

        return [
            ParamMapper(
                patterns="*",  # apply to all parameters
                at=self.n_epochs_frozen + 1,
                fn=unfreeze_parameter,
            )
        ]


Finetuning = Annotated[
    Union[LoRA, IA3, AdaLoRA, DoRA, OFT, FullFinetune, Frozen, TwoStages],
    Field(discriminator="kind"),
]
