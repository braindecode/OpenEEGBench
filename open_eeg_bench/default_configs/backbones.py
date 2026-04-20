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
        # w1, w2 are the two Linear layers inside FeedForward blocks
        peft_ff_modules=["w1", "w2"],
        normalization=PercentileScale(q=95.0),
        hub_repo="braindecode/biot-pretrained-prest-16chs",
        training_required_modules=["encoder.channel_tokens"],
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def labram(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.Labram",
        # mlp.0 and mlp.2 are the two Linear layers inside the MLP block
        peft_ff_modules=["mlp.0", "mlp.2"],
        normalization=DivideByConstant(factor=100.0),
        hub_repo="braindecode/labram-pretrained",
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def bendr(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.BENDR",
        # linear1, linear2 are the FFN layers in TransformerEncoderLayer
        peft_ff_modules=["linear1", "linear2"],
        training_required_modules=["channel_projection"],
        normalization=MinMaxScale(),
        hub_repo="braindecode/braindecode-bendr",
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def cbramod(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.CBraMod",
        # linear1, linear2 are the FFN layers in CrissCrossTransformerEncoderLayer
        peft_ff_modules=["linear1", "linear2"],
        normalization=DivideByConstant(factor=100.0),
        hub_repo="braindecode/cbramod-pretrained",
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def signal_jepa(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.SignalJEPA",
        # linear1, linear2 are the FFN layers in TransformerEncoderLayer
        peft_ff_modules=["linear1", "linear2"],
        hub_repo="braindecode/signal-jepa",
    )
    defaults.update(overrides)
    return PretrainedBackbone(**defaults)


def reve(**overrides) -> PretrainedBackbone:
    defaults = dict(
        model_cls="braindecode.models.REVE",
        # net.1, net.3 are the two Linear layers inside FeedForward.net
        peft_ff_modules=["net.1", "net.3"],
        normalization=WindowZScore(clip_sigma=15.0),
        hub_repo="brain-bzh/reve-base",
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
}
