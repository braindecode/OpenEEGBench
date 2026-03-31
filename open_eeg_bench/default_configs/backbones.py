"""Default backbone configurations for all supported architectures."""

from open_eeg_bench.backbone import PretrainedBackbone
from open_eeg_bench.normalization import (
    DivideByConstant,
    MinMaxScale,
    PercentileScale,
    WindowZScore,
)


def biot(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.BIOT",
        model_kwargs=dict(
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            drop_prob=0.5,
            att_drop_prob=0.2,
            att_layer_drop_prob=0.2,
            hop_length=100,
            max_seq_len=1024,
            return_feature=False,
        ),
        peft_target_modules=["to_q", "to_k", "to_v", "to_out", "w1", "w2"],
        peft_ff_modules=["w1", "w2"],
        normalization=PercentileScale(q=95.0),
        hub_repo="braindecode/biot-pretrained-prest-16chs",
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def labram(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.Labram",
        model_kwargs=dict(
            patch_size=200,
            embed_dim=200,
            num_layers=12,
            num_heads=10,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop_prob=0.1,
            attn_drop_prob=0.0,
            drop_path_prob=0.1,
            use_abs_pos_emb=True,
            use_mean_pooling=True,
            init_scale=0.001,
            init_values=0.1,
            neural_tokenizer=True,
        ),
        peft_target_modules=["qkv", "proj", "mlp.0", "mlp.2"],
        peft_ff_modules=["mlp.0", "mlp.2"],
        normalization=DivideByConstant(factor=100.0),
        hub_repo="braindecode/labram-pretrained",
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def bendr(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.BENDR",
        model_kwargs=dict(
            encoder_h=512,
            contextualizer_hidden=3076,
            transformer_layers=8,
            transformer_heads=8,
            position_encoder_length=25,
            enc_width=(3, 2, 2, 2, 2, 2),
            enc_downsample=(3, 2, 2, 2, 2, 2),
            drop_prob=0.1,
            layer_drop=0.0,
            projection_head=False,
            start_token=-5,
            final_layer=True,
        ),
        peft_target_modules=["out_proj", "linear1", "linear2"],
        peft_ff_modules=["linear1", "linear2"],
        normalization=MinMaxScale(),
        hub_repo="braindecode/braindecode-bendr",
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def cbramod(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.CBraMod",
        model_kwargs=dict(
            patch_size=200,
            dim_feedforward=800,
            n_layer=12,
            nhead=8,
            emb_dim=200,
            drop_prob=0.1,
            channels_kernel_stride_padding_norm=[
                [25, 49, 25, 24, [5, 25]],
                [25, 3, 1, 1, [5, 25]],
                [25, 3, 1, 1, [5, 25]],
            ],
            return_encoder_output=False,
        ),
        peft_target_modules=[
            "self_attn_s",
            "self_attn_t",
            "linear1",
            "linear2",
        ],
        peft_ff_modules=["linear1", "linear2"],
        normalization=DivideByConstant(factor=100.0),
        hub_repo="braindecode/cbramod-pretrained",
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def signal_jepa(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.SignalJEPA",
        model_kwargs=dict(
            feature_encoder__conv_layers_spec=[
                (8, 32, 8),
                (16, 2, 2),
                (32, 2, 2),
                (64, 2, 2),
                (64, 2, 2),
            ],
            feature_encoder__mode="default",
            feature_encoder__conv_bias=False,
            pos_encoder__spat_dim=30,
            pos_encoder__time_dim=34,
            pos_encoder__sfreq_features=1.0,
            transformer__d_model=64,
            transformer__num_encoder_layers=8,
            transformer__num_decoder_layers=4,
            transformer__nhead=8,
            drop_prob=0.0,
        ),
        peft_target_modules=[
            "linear1",
            "linear2",
            "self_attn.q_proj_weight",
            "self_attn.k_proj_weight",
            "self_attn.v_proj_weight",
        ],
        peft_ff_modules=["linear1", "linear2"],
        normalization=None,
        checkpoint_url="https://huggingface.co/braindecode/SignalJEPA/resolve/main/signal-jepa_16s-60_adeuwv4s.pth",
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def reve(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.REVE",
        model_kwargs=dict(
            embed_dim=512,
            depth=22,
            heads=8,
            head_dim=64,
            mlp_dim_ratio=2.66,
            use_geglu=True,
            patch_size=200,
            patch_overlap=20,
            attention_pooling=False,
        ),
        peft_target_modules=["to_qkv", "to_out", "net.1", "net.3"],
        peft_ff_modules=["net.1", "net.3"],
        normalization=WindowZScore(clip_sigma=15.0),
        hub_repo="brain-bzh/reve-base",
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def eegnet(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.EEGNet",
        model_kwargs=dict(
            F1=8,
            D=2,
            F2=16,
            kernel_length=64,
            depthwise_kernel_length=16,
            drop_prob=0.25,
        ),
        peft_target_modules=[],
        peft_ff_modules=[],
        head_module_name="final_layer",
        normalization=None,
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


ALL_BACKBONES = {
    "biot": biot,
    "labram": labram,
    "bendr": bendr,
    "cbramod": cbramod,
    "signal_jepa": signal_jepa,
    "reve": reve,
    "eegnet": eegnet,
}
