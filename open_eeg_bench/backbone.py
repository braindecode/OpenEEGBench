"""Backbone architecture configurations.

Each backbone class defines the architecture parameters, pretrained weights
source, and metadata needed by the finetuning and head modules
(target modules for PEFT, head module name, normalization, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

from open_eeg_bench.normalization import (
    DivideByConstant,
    MinMaxScale,
    Normalization,
    PercentileScale,
    WindowZScore,
)

log = logging.getLogger(__name__)

# Keys in the model kwargs that are metadata, not constructor params
_META_KEYS = {"kind", "hub_repo", "checkpoint_url"}


class _BackboneBase(BaseModel):
    """Base for all backbone configs.

    Subclasses must set class-level defaults for the coupling fields and
    implement ``_model_class`` and ``_model_kwargs``.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Coupling fields (queried by finetuning / head / dataset) ---
    peft_target_modules: list[str] = Field(
        description="Module names to target for PEFT adapters (LoRA, IA3, etc.)."
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

    # --- Pretrained weights ---
    hub_repo: str | None = Field(
        default=None, description="HuggingFace Hub repo for pretrained weights."
    )
    checkpoint_url: str | None = Field(
        default=None, description="Direct URL for pretrained weights."
    )

    def _model_class(self):
        raise NotImplementedError

    def _model_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the model constructor (excluding meta keys)."""
        raise NotImplementedError

    def build(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        sfreq: float,
        chs_info: list | None = None,
    ):
        """Instantiate the backbone model."""
        cls = self._model_class()
        kwargs = self._model_kwargs()
        kwargs.update(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            sfreq=sfreq,
            chs_info=chs_info,
        )
        return cls(**kwargs)

    def load_pretrained(self, model) -> None:
        """Load pretrained weights into the model."""
        if self.hub_repo is None and self.checkpoint_url is None:
            return

        import torch

        if self.hub_repo is not None:
            state_dict = self._load_from_hub(self.hub_repo)
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                self.checkpoint_url, progress=True
            )

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


# ============================================================================
# Concrete backbone configs
# ============================================================================


class BIOTBackbone(_BackboneBase):
    kind: Literal["biot"] = "biot"

    # Architecture
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    drop_prob: float = 0.5
    att_drop_prob: float = 0.2
    att_layer_drop_prob: float = 0.2
    hop_length: int = 100
    max_seq_len: int = 1024
    return_feature: bool = False

    # Coupling defaults
    peft_target_modules: list[str] = ["to_q", "to_k", "to_v", "to_out", "w1", "w2"]
    peft_ff_modules: list[str] = ["w1", "w2"]
    normalization: Normalization | None = PercentileScale(q=95.0)
    hub_repo: str | None = "braindecode/biot-pretrained-prest-16chs"

    def _model_class(self):
        from braindecode.models import BIOT

        return BIOT

    def _model_kwargs(self):
        return dict(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            drop_prob=self.drop_prob,
            att_drop_prob=self.att_drop_prob,
            att_layer_drop_prob=self.att_layer_drop_prob,
            hop_length=self.hop_length,
            max_seq_len=self.max_seq_len,
            return_feature=self.return_feature,
        )


class LabramBackbone(_BackboneBase):
    kind: Literal["labram"] = "labram"

    # Architecture
    patch_size: int = 200
    embed_dim: int = 200
    num_layers: int = 12
    num_heads: int = 10
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: float | None = None
    drop_prob: float = 0.1
    attn_drop_prob: float = 0.0
    drop_path_prob: float = 0.1
    use_abs_pos_emb: bool = True
    use_mean_pooling: bool = True
    init_scale: float = 0.001
    init_values: float = 0.1
    neural_tokenizer: bool = True

    # Coupling defaults
    peft_target_modules: list[str] = ["qkv", "proj", "mlp.0", "mlp.2"]
    peft_ff_modules: list[str] = ["mlp.0", "mlp.2"]
    normalization: Normalization | None = DivideByConstant(factor=100.0)
    hub_repo: str | None = "braindecode/labram-pretrained"

    def _model_class(self):
        from braindecode.models import Labram

        return Labram

    def _model_kwargs(self):
        return dict(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            drop_prob=self.drop_prob,
            attn_drop_prob=self.attn_drop_prob,
            drop_path_prob=self.drop_path_prob,
            use_abs_pos_emb=self.use_abs_pos_emb,
            use_mean_pooling=self.use_mean_pooling,
            init_scale=self.init_scale,
            init_values=self.init_values,
            neural_tokenizer=self.neural_tokenizer,
        )


class BENDRBackbone(_BackboneBase):
    kind: Literal["bendr"] = "bendr"

    # Architecture
    encoder_h: int = 512
    contextualizer_hidden: int = 3076
    transformer_layers: int = 8
    transformer_heads: int = 8
    position_encoder_length: int = 25
    enc_width: tuple[int, ...] = (3, 2, 2, 2, 2, 2)
    enc_downsample: tuple[int, ...] = (3, 2, 2, 2, 2, 2)
    drop_prob: float = 0.1
    layer_drop: float = 0.0
    projection_head: bool = False
    start_token: int = -5
    final_layer: bool = True

    # Coupling defaults
    peft_target_modules: list[str] = ["out_proj", "linear1", "linear2"]
    peft_ff_modules: list[str] = ["linear1", "linear2"]
    normalization: Normalization | None = MinMaxScale()
    hub_repo: str | None = "braindecode/braindecode-bendr"

    def _model_class(self):
        from braindecode.models import BENDR

        return BENDR

    def _model_kwargs(self):
        return dict(
            encoder_h=self.encoder_h,
            contextualizer_hidden=self.contextualizer_hidden,
            transformer_layers=self.transformer_layers,
            transformer_heads=self.transformer_heads,
            position_encoder_length=self.position_encoder_length,
            enc_width=self.enc_width,
            enc_downsample=self.enc_downsample,
            drop_prob=self.drop_prob,
            layer_drop=self.layer_drop,
            projection_head=self.projection_head,
            start_token=self.start_token,
            final_layer=self.final_layer,
        )


class CBraModBackbone(_BackboneBase):
    kind: Literal["cbramod"] = "cbramod"

    # Architecture
    patch_size: int = 200
    dim_feedforward: int = 800
    n_layer: int = 12
    nhead: int = 8
    emb_dim: int = 200
    drop_prob: float = 0.1
    channels_kernel_stride_padding_norm: list[list] = Field(
        default=[
            [25, 49, 25, 24, [5, 25]],
            [25, 3, 1, 1, [5, 25]],
            [25, 3, 1, 1, [5, 25]],
        ]
    )
    return_encoder_output: bool = False

    # Coupling defaults
    peft_target_modules: list[str] = [
        "self_attn_s",
        "self_attn_t",
        "linear1",
        "linear2",
    ]
    peft_ff_modules: list[str] = ["linear1", "linear2"]
    normalization: Normalization | None = DivideByConstant(factor=100.0)
    hub_repo: str | None = "braindecode/cbramod-pretrained"

    def _model_class(self):
        from braindecode.models import CBraMod

        return CBraMod

    def _model_kwargs(self):
        return dict(
            patch_size=self.patch_size,
            dim_feedforward=self.dim_feedforward,
            n_layer=self.n_layer,
            nhead=self.nhead,
            emb_dim=self.emb_dim,
            drop_prob=self.drop_prob,
            channels_kernel_stride_padding_norm=self.channels_kernel_stride_padding_norm,
            return_encoder_output=self.return_encoder_output,
        )


class SignalJEPABackbone(_BackboneBase):
    kind: Literal["signal_jepa"] = "signal_jepa"

    # Architecture
    feature_encoder__conv_layers_spec: list[list[int]] = Field(
        default=[[8, 32, 8], [16, 2, 2], [32, 2, 2], [64, 2, 2], [64, 2, 2]]
    )
    feature_encoder__mode: str = "default"
    feature_encoder__conv_bias: bool = False
    pos_encoder__spat_dim: int = 30
    pos_encoder__time_dim: int = 34
    pos_encoder__sfreq_features: float = 1.0
    transformer__d_model: int = 64
    transformer__num_encoder_layers: int = 8
    transformer__num_decoder_layers: int = 4
    transformer__nhead: int = 8
    drop_prob: float = 0.0

    # Coupling defaults
    peft_target_modules: list[str] = [
        "linear1",
        "linear2",
        "self_attn.q_proj_weight",
        "self_attn.k_proj_weight",
        "self_attn.v_proj_weight",
    ]
    peft_ff_modules: list[str] = ["linear1", "linear2"]
    normalization: Normalization | None = None
    checkpoint_url: str | None = "https://huggingface.co/braindecode/SignalJEPA/resolve/main/signal-jepa_16s-60_adeuwv4s.pth"

    def _model_class(self):
        from braindecode.models import SignalJEPA

        return SignalJEPA

    def _model_kwargs(self):
        return {
            "feature_encoder__conv_layers_spec": [
                tuple(x) for x in self.feature_encoder__conv_layers_spec
            ],
            "feature_encoder__mode": self.feature_encoder__mode,
            "feature_encoder__conv_bias": self.feature_encoder__conv_bias,
            "pos_encoder__spat_dim": self.pos_encoder__spat_dim,
            "pos_encoder__time_dim": self.pos_encoder__time_dim,
            "pos_encoder__sfreq_features": self.pos_encoder__sfreq_features,
            "transformer__d_model": self.transformer__d_model,
            "transformer__num_encoder_layers": self.transformer__num_encoder_layers,
            "transformer__num_decoder_layers": self.transformer__num_decoder_layers,
            "transformer__nhead": self.transformer__nhead,
            "drop_prob": self.drop_prob,
        }


class REVEBackbone(_BackboneBase):
    kind: Literal["reve"] = "reve"

    # Architecture
    embed_dim: int = 512
    depth: int = 22
    heads: int = 8
    head_dim: int = 64
    mlp_dim_ratio: float = 2.66
    use_geglu: bool = True
    patch_size: int = 200
    patch_overlap: int = 20
    attention_pooling: bool = False

    # Coupling defaults
    peft_target_modules: list[str] = ["to_qkv", "to_out", "net.1", "net.3"]
    peft_ff_modules: list[str] = ["net.1", "net.3"]
    normalization: Normalization | None = WindowZScore(clip_sigma=15.0)
    hub_repo: str | None = "brain-bzh/reve-base"

    def _model_class(self):
        from braindecode.models import REVE

        return REVE

    def _model_kwargs(self):
        return dict(
            embed_dim=self.embed_dim,
            depth=self.depth,
            heads=self.heads,
            head_dim=self.head_dim,
            mlp_dim_ratio=self.mlp_dim_ratio,
            use_geglu=self.use_geglu,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
            attention_pooling=self.attention_pooling,
        )


Backbone = Annotated[
    Union[
        BIOTBackbone,
        LabramBackbone,
        BENDRBackbone,
        CBraModBackbone,
        SignalJEPABackbone,
        REVEBackbone,
    ],
    Field(discriminator="kind"),
]
