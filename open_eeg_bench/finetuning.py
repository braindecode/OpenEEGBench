"""Fine-tuning strategy configurations.

Each strategy defines what to freeze/adapt and how.  PEFT-based strategies
(LoRA, IA3, …) wrap the model with HuggingFace PEFT.  ``Frozen`` freezes
the encoder and returns a skorch ``Freezer`` callback.  ``FullFinetune``
trains everything.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


log = logging.getLogger(__name__)


def _param_stats(model) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }


def _nn_types(*names: str) -> tuple:
    """Return a tuple of ``torch.nn`` classes by name (lazy import)."""
    import torch.nn as nn

    return tuple(getattr(nn, n) for n in names)


def _disable_dropout(model):
    import torch.nn as nn

    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
    )
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.p = 0.0


def _filter_targets(
    model, target_modules: list[str] | Literal["all-linear"] | None, supported: tuple
) -> list[str] | Literal["all-linear"] | None:
    """Keep only *target_modules* whose underlying layer is an instance of *supported*."""
    if target_modules is None or target_modules == "all-linear":
        return target_modules
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
        log.warning(
            "Filtered out unsupported target modules: %s (keeping: %s)",
            sorted(filtered_out),
            sorted(valid),
        )
    return sorted(valid)


def _apply_peft(model, peft_config):
    """Apply PEFT config to the model."""
    from peft import get_peft_model

    peft_model = get_peft_model(model, peft_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    return peft_model, trainable, total


class _BaseFinetuning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disable_backbone_dropout: bool = True

    def _apply(self, model, backbone):
        raise NotImplementedError

    def apply(self, model, backbone):
        if self.disable_backbone_dropout:
            _disable_dropout(model)
        wrapped, stats = self._apply(model, backbone)
        return wrapped, stats


# ============================================================================
# Concrete strategies
# ============================================================================


class LoRA(_BaseFinetuning):
    """Low-Rank Adaptation."""

    kind: Literal["lora"] = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    bias: Literal["none", "all", "lora_only"] = "none"

    def _apply(self, model, backbone):
        from peft import LoraConfig as PeftLoraConfig

        modules_to_save = backbone.get_training_required_modules()
        target_modules = _filter_targets(
            model,
            backbone.peft_target_modules,
            _nn_types(
                "Linear",
                "Conv1d",
                "Conv2d",
                "Conv3d",
                "Embedding",
                "MultiheadAttention",
            ),
        )
        cfg = PeftLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=target_modules,
            bias=self.bias,
            modules_to_save=modules_to_save,
        )
        wrapped, trainable, total = _apply_peft(model, cfg)
        return wrapped, {
            "method": "lora",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

    def get_callbacks(self) -> list:
        return []


class IA3(_BaseFinetuning):
    """Infused Adapter by Inhibiting and Amplifying Inner Activations."""

    kind: Literal["ia3"] = "ia3"

    def _apply(self, model, backbone):
        from peft import IA3Config as PeftIA3Config

        modules_to_save = backbone.get_training_required_modules()
        ff_modules = backbone.peft_ff_modules or None
        target_modules = backbone.peft_target_modules
        if ff_modules and isinstance(target_modules, list):
            target_modules = list(set(target_modules) | set(ff_modules))
        target_modules = _filter_targets(
            model, target_modules, _nn_types("Linear", "Conv2d", "Conv3d")
        )
        if ff_modules and isinstance(target_modules, list):
            ff_modules = [m for m in ff_modules if m in target_modules]
        cfg = PeftIA3Config(
            target_modules=target_modules,
            feedforward_modules=ff_modules or None,
            modules_to_save=modules_to_save,
        )
        wrapped, trainable, total = _apply_peft(model, cfg)
        return wrapped, {
            "method": "ia3",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

    def get_callbacks(self) -> list:
        return []


class AdaLoRA(_BaseFinetuning):
    """Adaptive Low-Rank Adaptation with dynamic rank allocation."""

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

    def _apply(self, model, backbone):
        from peft import AdaLoraConfig as PeftAdaLoraConfig

        modules_to_save = backbone.get_training_required_modules()
        target_modules = _filter_targets(
            model, backbone.peft_target_modules, _nn_types("Linear")
        )
        cfg = PeftAdaLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            total_step=self.total_step,
            target_r=self.target_r,
            init_r=self.init_r,
            tinit=self.tinit,
            tfinal=self.tfinal,
            deltaT=self.deltaT,
        )
        wrapped, trainable, total = _apply_peft(model, cfg)
        return wrapped, {
            "method": "adalora",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

    def get_callbacks(self) -> list:
        return []


class DoRA(_BaseFinetuning):
    """Weight-Decomposed Low-Rank Adaptation."""

    kind: Literal["dora"] = "dora"
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    bias: Literal["none", "all", "lora_only"] = "none"

    def _apply(self, model, backbone):
        from peft import LoraConfig as PeftLoraConfig

        modules_to_save = backbone.get_training_required_modules()
        target_modules = _filter_targets(
            model,
            backbone.peft_target_modules,
            _nn_types("Linear", "Conv1d", "Conv2d", "Conv3d", "Embedding"),
        )
        cfg = PeftLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=target_modules,
            bias=self.bias,
            modules_to_save=modules_to_save,
            use_dora=True,
        )
        wrapped, trainable, total = _apply_peft(model, cfg)
        return wrapped, {
            "method": "dora",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

    def get_callbacks(self) -> list:
        return []


class OFT(_BaseFinetuning):
    """Orthogonal Fine-Tuning."""

    kind: Literal["oft"] = "oft"
    block_size: int = 8
    module_dropout: float = 0.0
    coft: bool = False
    block_share: bool = False

    def _apply(self, model, backbone):
        from peft import OFTConfig as PeftOFTConfig

        modules_to_save = backbone.get_training_required_modules()
        target_modules = _filter_targets(
            model, backbone.peft_target_modules, _nn_types("Linear", "Conv2d")
        )
        cfg = PeftOFTConfig(
            oft_block_size=self.block_size,
            target_modules=target_modules,
            module_dropout=self.module_dropout,
            coft=self.coft,
            block_share=self.block_share,
            modules_to_save=modules_to_save,
        )
        wrapped, trainable, total = _apply_peft(model, cfg)
        return wrapped, {
            "method": "oft",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

    def get_callbacks(self) -> list:
        return []


class FullFinetune(_BaseFinetuning):
    """Train all parameters (no adapters)."""

    kind: Literal["full"] = "full"

    def _apply(self, model, backbone):
        return model, {**_param_stats(model), "method": "full"}

    def get_callbacks(self) -> list:
        return []


class Frozen(_BaseFinetuning):
    """Freeze the encoder, train only the head.

    Returns a model in which all parameters
    whose name does *not* contain the head module name are frozen.
    """

    kind: Literal["frozen"] = "frozen"

    def _apply(self, model, backbone):
        modules_to_save = backbone.get_training_required_modules()
        for name, param in model.named_parameters():
            if not any(save_name in name for save_name in modules_to_save):
                param.requires_grad = False
        return model, {**_param_stats(model), "method": "frozen"}

    def get_callbacks(self) -> list:
        return []


class TwoStages(_BaseFinetuning):
    """Two-stage training: frozen backbone for N epochs, then unfreeze and train all.

    Returns a model in which all parameters
    whose name does *not* contain the head module name are frozen.
    After N epochs, an Unfreezer callback will unfreeze the whole model for the rest of training.
    """

    kind: Literal["two_stages"] = "two_stages"
    n_epochs_frozen: int = 5

    def _apply(self, model, backbone):
        modules_to_save = backbone.get_training_required_modules()
        for name, param in model.named_parameters():
            if not any(save_name in name for save_name in modules_to_save):
                param.requires_grad = False
        # TODO: set the rest of the modules to eval mode!
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
