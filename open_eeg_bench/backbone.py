"""Backbone architecture configurations.

``_BackboneBase`` defines the coupling fields and ``build()`` wrapper shared
by every backbone config.  ``_BraindecodeBackbone`` adds the braindecode-style
``model_cls`` + ``model_kwargs`` instantiation logic used by the two concrete
implementations:

* ``PretrainedBackbone`` — loads pretrained weights from HF Hub, a URL, or a
  local checkpoint.
* ``ScratchBackbone`` — trains from random initialization (no weights loaded).

Default configurations for each supported architecture live in
``default_configs.backbones``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal
from importlib import import_module

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_eeg_bench.normalization import NoNormalization, Normalization

log = logging.getLogger(__name__)


class _BackboneBase(BaseModel):
    """Base for all backbone configs."""

    model_config = ConfigDict(extra="forbid")

    # --- Coupling fields (queried by finetuning / head / dataset) ---
    peft_target_modules: list[str] | Literal["all-linear"] | None = Field(
        default="all-linear",
        description="Module names to target for PEFT adapters (LoRA, IA3, etc.).",
    )
    peft_ff_modules: list[str] | None = Field(
        default=None,
        description="Feedforward module names for IA3.",
    )
    head_module_name: str = Field(
        default="final_layer",
        description=(
            "Name of the classification head module in the model. "
            "This layer will be discarded and replaced with a new head during finetuning."
        ),
    )
    training_required_modules: list[str] | None = Field(
        default=None,
        description=(
            "Module names to always train, even in 'frozen' finetuning mode "
            "(where by default only final_layer is trained). "
            "Leave empty unless the backbone needs an adaptation layer for unseen channels. "
            "Warning: models using this field are categorized separately in evaluations, "
            "since training extra layers is not comparable to training only a classification head."
        ),
    )
    normalization: Normalization = Field(
        default=NoNormalization(),
        description="Post-window normalization applied to each data window.",
    )

    def get_training_required_modules(self):
        out = [self.head_module_name]
        if self.training_required_modules:
            log.warning(
                "%s requires training extra modules %s beyond the head. "
                "This model will be categorized separately in evaluations.",
                type(self).__name__,
                self.training_required_modules,
            )
            out += self.training_required_modules
        return out

    def _build(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        sfreq: float,
        chs_info: list | None = None,
    ):
        """Instantiate the backbone model."""
        raise NotImplementedError

    def _check_layers_and_parameters_exist(self, model):
        """Verify that all configured module/parameter names exist in the model.

        This test can only be done once the model has been instantiated.
        """
        module_names = {name for name, _ in model.named_modules()}
        module_fields = {"head_module_name": [self.head_module_name]}
        if (
            self.peft_target_modules is not None
            and self.peft_target_modules != "all-linear"
        ):
            module_fields["peft_target_modules"] = self.peft_target_modules
        if self.peft_ff_modules is not None:
            module_fields["peft_ff_modules"] = self.peft_ff_modules
        if self.training_required_modules is not None:
            module_fields["training_required_modules"] = self.training_required_modules
        for field, names in module_fields.items():
            for name in names:
                # Allow short names that match a suffix (e.g. "qkv" matches "encoder.layer.0.qkv")
                if name not in module_names and not any(
                    n == name or n.endswith("." + name) for n in module_names
                ):
                    raise ValueError(
                        f"{type(self).__name__}.{field} references module '{name}' "
                        f"which does not exist in the model."
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
        model = self._build(n_chans, n_times, n_outputs, sfreq, chs_info)
        self._check_layers_and_parameters_exist(model)
        return model


# ============================================================================
# Concrete backbone configs
# ============================================================================


class _BraindecodeBackbone(_BackboneBase):
    """Base for backbones backed by any braindecode-compatible model class.

    Parameters
    ----------
    model_cls : str
        Dotted import path to the model class,
        e.g. ``"braindecode.models.BIOT"``. Resolved lazily in ``build()``.
    model_kwargs : dict
        Keyword arguments forwarded to the model constructor
        (architecture hyperparameters).
    """

    model_cls: str
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

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

    def _build(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        sfreq: float,
        chs_info: list | None = None,
    ):
        """Instantiate the backbone model (without loading any weights)."""
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


class PretrainedBackbone(_BraindecodeBackbone):
    """Braindecode-compatible backbone initialized from pretrained weights.

    Parameters
    ----------
    hub_repo : str, optional
        HuggingFace Hub repo ID for pretrained weights, e.g. ``"braindecode/biot-pretrained-prest-16chs"``.
    checkpoint_url : str, optional
        Direct URL for pretrained weights (e.g. a .bin or .safetensors file).
    checkpoint_path : str, optional
        Local filesystem path to pretrained weights (.pth, .bin, or .safetensors).
        Supports both raw state dicts and checkpoint dicts with a ``"state_dict"`` key.
    """

    kind: Literal["hf_hub"] = "hf_hub"

    hub_repo: str | None = None
    checkpoint_url: str | None = None
    checkpoint_path: str | None = None

    def _build(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        sfreq: float,
        chs_info: list | None = None,
    ):
        """Instantiate the backbone model and load pretrained weights."""
        model = super()._build(n_chans, n_times, n_outputs, sfreq, chs_info)
        self.load_pretrained(model)
        return model

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
        if (
            self.hub_repo is None
            and self.checkpoint_url is None
            and self.checkpoint_path is None
        ):
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

        allowed_modules = [self.head_module_name] + (
            self.training_required_modules or []
        )

        def covered(k: str) -> bool:
            return any(f".{m}." in f".{k}." for m in allowed_modules)

        def describe(k: str) -> str:
            if k in state_dict:
                ckpt_shape = tuple(state_dict[k].shape)
                model_shape = tuple(model_state[k].shape)
                return f"{k}  [shape mismatch: ckpt {ckpt_shape} vs model {model_shape}]"
            return f"{k}  [absent from checkpoint]"

        param_names = {n for n, _ in model.named_parameters()}
        missing_params = [k for k in missing if k in param_names and not covered(k)]
        missing_buffers = [
            k for k in missing if k not in param_names and not covered(k)
        ]

        if missing_params:
            raise ValueError(
                f"Pretrained checkpoint for {self.model_cls} is missing "
                f"{len(missing_params)} learnable parameter(s) that are neither "
                f"under head_module_name='{self.head_module_name}' nor under any "
                f"module in training_required_modules={self.training_required_modules}:\n"
                + "\n".join(f"  - {describe(k)}" for k in missing_params)
                + "\nThese parameters would be silently initialized from scratch, "
                "which is almost certainly a config error. Either:\n"
                "  (a) the checkpoint is incompatible with the declared architecture "
                "(check model_kwargs), or\n"
                "  (b) these modules should be declared in `training_required_modules` "
                "so they are explicitly trained from scratch (note: this will "
                "categorize the model separately in evaluations)."
            )

        if missing_buffers:
            log.warning(
                "Pretrained checkpoint for %s is missing %d buffer(s) that are "
                "neither under head_module_name='%s' nor under any module in "
                "training_required_modules=%s:\n%s\n"
                "Buffers are not trained, so missing values may be computed at "
                "init — but verify this is intentional.",
                self.model_cls,
                len(missing_buffers),
                self.head_module_name,
                self.training_required_modules,
                "\n".join(f"  - {describe(k)}" for k in missing_buffers),
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


class ScratchBackbone(_BraindecodeBackbone):
    """Braindecode-compatible backbone trained from random initialization.

    Unlike :class:`PretrainedBackbone`, no weights are loaded — the model is
    instantiated and trained from scratch.  Only compatible with the
    ``FullFinetune`` strategy (enforced by the ``Experiment`` validator).
    """

    kind: Literal["scratch"] = "scratch"


class PlaceholderBackbone(_BackboneBase):
    """Placeholder backbone for building experiment configs without choosing a model yet.

    Cannot be used to actually run an experiment — the Experiment validator
    will reject it.
    """

    kind: Literal["placeholder"] = "placeholder"
