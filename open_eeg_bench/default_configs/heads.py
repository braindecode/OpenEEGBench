"""Default head configurations."""

from open_eeg_bench.head import LinearHead, MLPHead, OriginalHead


def linear_head(**overrides) -> LinearHead:
    return LinearHead(**overrides)


def mlp_head(**overrides) -> MLPHead:
    return MLPHead(**overrides)


def original_head(**overrides) -> OriginalHead:
    return OriginalHead(**overrides)


ALL_HEADS = [linear_head, mlp_head, original_head]
