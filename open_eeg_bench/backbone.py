"""Backbone architecture configurations.

``_BackboneBase`` defines the interface shared by every backbone config.
``PretrainedBackbone`` is the concrete implementation: it takes an
``ImportString`` pointing to any braindecode (or compatible) model class
plus free-form constructor kwargs.

Default configurations for each supported architecture live in
``default_configs.backbones``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Union

from importlib import import_module

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

from open_eeg_bench.normalization import Normalization

log = logging.getLogger(__name__)


class _BackboneBase(BaseModel):
    """Base for all backbone configs."""

    model_config = ConfigDict(extra="forbid")

    # --- Coupling fields (queried by finetuning / head / dataset) ---
    peft_target_modules: list[str] = Field(
        default_factory=list,
        description="Module names to target for PEFT adapters (LoRA, IA3, etc.).",
    )
    peft_ff_modules: list[str] = Field(
        default_factory=list,
        description="Feedforward module names for IA3.",
    )
    head_module_name: str = Field(
        default="final_layer",
        description="Name of the classification head module in the model.",
    )
    normalization: Normalization | None = Field(
        default=None,
        description="Post-window normalization applied to each data window.",
    )

    def build(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        sfreq: float,
        chs_info: list | None = None,
    ):
        """Instantiate the backbone model."""
        raise NotImplementedError


# ============================================================================
# Concrete backbone configs
# ============================================================================


class PretrainedBackbone(_BackboneBase):
    """Generic backbone backed by any braindecode-compatible model class.

    Parameters
    ----------
    model_cls : str
        Dotted import path to the model class,
        e.g. ``"braindecode.models.BIOT"``. Resolved lazily in ``build()``.
    model_kwargs : dict
        Keyword arguments forwarded to the model constructor
        (architecture hyperparameters).
    hub_repo : str, optional
        HuggingFace Hub repo ID for pretrained weights, e.g. ``"braindecode/biot-pretrained-prest-16chs"``.
    checkpoint_url : str, optional
        Direct URL for pretrained weights (e.g. a .bin or .safetensors file).
    checkpoint_path : str, optional
        Local filesystem path to pretrained weights (.pth, .bin, or .safetensors).
        Supports both raw state dicts and checkpoint dicts with a ``"state_dict"`` key.
    """

    kind: Literal["hf_hub"] = "hf_hub"

    model_cls: str
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    hub_repo: str | None = None
    checkpoint_url: str | None = None
    checkpoint_path: str | None = None

    def _resolve_model_cls(self):
        """Resolve and return the model class from a dotted import path.
        Dynamically imports a model class specified as a fully-qualified dotted path
        (e.g., 'path.to.module.ClassName'). This approach avoids using pydantic.ImportString
        to prevent expensive imports (torch, braindecode) from being triggered during
        validation, which is critical for resource-constrained environments like cluster
        login nodes.
        """
        module_path, cls_name = self.model_cls.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, cls_name)

    def build(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        sfreq: float,
        chs_info: list | None = None,
    ):
        """Instantiate the backbone model."""
        cls = self._resolve_model_cls()
        kwargs = self.model_kwargs.copy()
        kwargs.update(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            sfreq=sfreq,
            chs_info=chs_info,
        )
        return cls(**kwargs)

    @model_validator(mode="after")
    def check_pretrained_fields(self):
        sources = [
            self.hub_repo is not None,
            self.checkpoint_url is not None,
            self.checkpoint_path is not None,
        ]
        if sum(sources) != 1:
            raise ValueError(
                "Exactly one of hub_repo, checkpoint_url, or checkpoint_path "
                "must be provided."
            )
        return self

    def load_pretrained(self, model) -> None:
        """Load pretrained weights into the model."""
        if self.hub_repo is None and self.checkpoint_url is None and self.checkpoint_path is None:
            return

        import torch

        if self.hub_repo is not None:
            state_dict = self._load_from_hub(self.hub_repo)
        elif self.checkpoint_url is not None:
            state_dict = torch.hub.load_state_dict_from_url(
                self.checkpoint_url, progress=True
            )
        elif self.checkpoint_path is not None:
            state_dict = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        else:
            raise ValueError("No pretrained weights specified.")

        # Filter out incompatible keys (shape mismatch or missing)
        model_state = model.state_dict()
        filtered, skipped = {}, []
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                reason = (
                    f"shape {v.shape} vs {model_state[k].shape}"
                    if k in model_state
                    else "not in model"
                )
                skipped.append(f"{k} ({reason})")

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        log.info(
            "Pretrained: loaded %d/%d keys, %d missing, %d skipped",
            len(filtered),
            len(state_dict),
            len(missing),
            len(skipped),
        )

    @staticmethod
    def _load_from_hub(repo_id: str) -> dict:
        import torch
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
            import safetensors.torch

            return safetensors.torch.load_file(path)
        except Exception:
            path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            return torch.load(path, map_location="cpu", weights_only=False)


class PlaceholderBackbone(_BackboneBase):
    """Placeholder backbone for building experiment configs without choosing a model yet.

    Cannot be used to actually run an experiment — the Experiment validator
    will reject it.
    """

    kind: Literal["placeholder"] = "placeholder"
    peft_target_modules: list[str] = []

