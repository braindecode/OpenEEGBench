"""Fine-tuning strategy configurations.

Each strategy defines what to freeze/adapt and how.  PEFT-based strategies
(LoRA, IA3, …) wrap the model with HuggingFace PEFT.  ``Frozen`` freezes
the encoder and returns a skorch ``Freezer`` callback.  ``FullFinetune``
trains everything.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

import torch.nn as nn
from peft import (
    AdaLoraConfig as PeftAdaLoraConfig,
    IA3Config as PeftIA3Config,
    LoraConfig as PeftLoraConfig,
    OFTConfig as PeftOFTConfig,
    get_peft_model,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from open_eeg_bench.backbone import _BackboneBase

log = logging.getLogger(__name__)


class _PeftModelWrapper(nn.Module):
    """Wraps a PEFT model so forward() calls the base model directly.

    PEFT's own forward expects HuggingFace-style models with a ``config``
    attribute.  This wrapper bypasses that.
    """

    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model

    def forward(self, *args, **kwargs):
        return self.peft_model.base_model.model(*args, **kwargs)

    def __getattr__(self, name: str):
        if name in ("peft_model", "training"):
            return super().__getattr__(name)
        try:
            return getattr(self.peft_model, name)
        except AttributeError:
            return getattr(self.peft_model.base_model.model, name)

    def train(self, mode: bool = True):
        super().train(mode)
        self.peft_model.train(mode)
        return self

    def eval(self):
        return self.train(False)


def _param_stats(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }


def _resolve_modules_to_save(model: nn.Module, head_module_name: str) -> list[str] | None:
    """Auto-detect the head module to keep trainable under PEFT."""
    if hasattr(model, head_module_name):
        return [head_module_name]
    for candidate in ("final_layer", "classifier", "head"):
        if hasattr(model, candidate):
            return [candidate]
    return None


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

    def apply(self, model: nn.Module, backbone: _BackboneBase) -> tuple[nn.Module, dict]:
        modules_to_save = _resolve_modules_to_save(model, backbone.head_module_name)
        cfg = PeftLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=backbone.peft_target_modules,
            bias=self.bias,
            modules_to_save=modules_to_save,
        )
        peft_model = get_peft_model(model, cfg)
        trainable, total = peft_model.get_nb_trainable_parameters()
        wrapped = _PeftModelWrapper(peft_model)
        return wrapped, {
            "method": "lora",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

    def get_callbacks(self) -> list:
        return []


class IA3(BaseModel):
    """Infused Adapter by Inhibiting and Amplifying Inner Activations."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["ia3"] = "ia3"

    def apply(self, model: nn.Module, backbone: _BackboneBase) -> tuple[nn.Module, dict]:
        modules_to_save = _resolve_modules_to_save(model, backbone.head_module_name)
        ff_modules = backbone.peft_ff_modules or None
        target_modules = list(backbone.peft_target_modules)
        if ff_modules:
            target_modules = list(set(target_modules) | set(ff_modules))
        cfg = PeftIA3Config(
            target_modules=target_modules,
            feedforward_modules=ff_modules,
            modules_to_save=modules_to_save,
        )
        peft_model = get_peft_model(model, cfg)
        trainable, total = peft_model.get_nb_trainable_parameters()
        wrapped = _PeftModelWrapper(peft_model)
        return wrapped, {
            "method": "ia3",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

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

    def apply(self, model: nn.Module, backbone: _BackboneBase) -> tuple[nn.Module, dict]:
        cfg = PeftAdaLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=backbone.peft_target_modules,
            total_step=self.total_step,
            target_r=self.target_r,
            init_r=self.init_r,
            tinit=self.tinit,
            tfinal=self.tfinal,
            deltaT=self.deltaT,
        )
        peft_model = get_peft_model(model, cfg)
        trainable, total = peft_model.get_nb_trainable_parameters()
        wrapped = _PeftModelWrapper(peft_model)
        return wrapped, {
            "method": "adalora",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

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

    def apply(self, model: nn.Module, backbone: _BackboneBase) -> tuple[nn.Module, dict]:
        modules_to_save = _resolve_modules_to_save(model, backbone.head_module_name)
        cfg = PeftLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=backbone.peft_target_modules,
            bias=self.bias,
            modules_to_save=modules_to_save,
            use_dora=True,
        )
        peft_model = get_peft_model(model, cfg)
        trainable, total = peft_model.get_nb_trainable_parameters()
        wrapped = _PeftModelWrapper(peft_model)
        return wrapped, {
            "method": "dora",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

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

    def apply(self, model: nn.Module, backbone: _BackboneBase) -> tuple[nn.Module, dict]:
        modules_to_save = _resolve_modules_to_save(model, backbone.head_module_name)
        cfg = PeftOFTConfig(
            oft_block_size=self.block_size,
            target_modules=backbone.peft_target_modules,
            module_dropout=self.module_dropout,
            coft=self.coft,
            block_share=self.block_share,
            modules_to_save=modules_to_save,
        )
        peft_model = get_peft_model(model, cfg)
        trainable, total = peft_model.get_nb_trainable_parameters()
        wrapped = _PeftModelWrapper(peft_model)
        return wrapped, {
            "method": "oft",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

    def get_callbacks(self) -> list:
        return []


class FullFinetune(BaseModel):
    """Train all parameters (no adapters)."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["full"] = "full"

    def apply(self, model: nn.Module, backbone: _BackboneBase) -> tuple[nn.Module, dict]:
        return model, {**_param_stats(model), "method": "full"}

    def get_callbacks(self) -> list:
        return []


class Frozen(BaseModel):
    """Freeze the encoder, train only the head.

    Returns a skorch ``Freezer`` callback that freezes all parameters
    whose name does *not* contain the head module name.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["frozen"] = "frozen"

    def apply(self, model: nn.Module, backbone: _BackboneBase) -> tuple[nn.Module, dict]:
        head_name = backbone.head_module_name
        for name, param in model.named_parameters():
            if head_name not in name:
                param.requires_grad = False
        return model, {**_param_stats(model), "method": "frozen"}

    def get_callbacks(self) -> list:
        return []


Finetuning = Annotated[
    Union[LoRA, IA3, AdaLoRA, DoRA, OFT, FullFinetune, Frozen],
    Field(discriminator="kind"),
]
