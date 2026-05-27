"""Test normalization transforms."""

import numpy as np
import pytest
from pydantic import BaseModel

from open_eeg_bench.normalization import (
    DivideByConstant,
    MinMaxScale,
    NoNormalization,
    Normalization,
    PercentileScale,
    ScaleToMV,
    WindowZScore,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((19, 1000)).astype(np.float32) * 100


def test_divide_by_constant(data):
    norm = DivideByConstant(factor=100.0)
    result = norm.apply(data)
    np.testing.assert_allclose(result, data / 100.0)


def test_percentile_scale(data):
    norm = PercentileScale(q=95.0)
    result = norm.apply(data)
    assert result.shape == data.shape
    assert np.abs(result).max() < 10  # most values near [-1, 1]


def test_minmax_scale(data):
    norm = MinMaxScale()
    result = norm.apply(data)
    assert result.min() >= -1.0 - 1e-6
    assert result.max() <= 1.0 + 1e-6


def test_window_zscore(data):
    norm = WindowZScore(clip_sigma=15.0)
    result = norm.apply(data)
    assert result.shape == data.shape
    assert np.abs(result).max() <= 15.0 + 1e-6


def test_window_zscore_channelwise(data):
    norm = WindowZScore(channel_wise=True, clip_sigma=None)
    result = norm.apply(data)
    # Each channel should be ~zero mean, ~unit std
    for ch in range(data.shape[0]):
        assert abs(np.mean(result[ch])) < 0.01
        assert abs(np.std(result[ch]) - 1.0) < 0.01


def test_scale_to_mv(data):
    norm = ScaleToMV()
    result = norm.apply(data)
    np.testing.assert_allclose(result, data / 1000.0)


def test_custom_normalization(data):
    """A user-defined Normalization subclass should integrate seamlessly."""

    class AddOffset(Normalization):
        offset: float = 1.0

        def apply(self, data: np.ndarray) -> np.ndarray:
            return data + self.offset

    # Direct use
    norm = AddOffset(offset=2.5)
    np.testing.assert_allclose(norm.apply(data), data + 2.5)

    # Usable as a Normalization field on a parent model, with round-trip
    # serialization through the "kind" discriminator key.
    class Parent(BaseModel):
        norm: Normalization

    parent = Parent(norm=AddOffset(offset=3.0))
    assert isinstance(parent.norm, AddOffset)

    restored = Parent.model_validate(parent.model_dump())
    assert isinstance(restored.norm, AddOffset)
    assert restored.norm.offset == 3.0


@pytest.mark.parametrize(
    "cls, legacy_kind",
    [
        (DivideByConstant, "divide_by_constant"),
        (PercentileScale, "percentile_scale"),
        (MinMaxScale, "minmax_scale"),
        (WindowZScore, "window_zscore"),
        (ScaleToMV, "scale_to_mv"),
        (NoNormalization, "none"),
    ],
)
def test_legacy_kind_serialization(cls, legacy_kind):
    """Builtin subclasses must keep their pre-DiscriminatedModel ``kind`` value
    so cached experiment UIDs remain stable."""
    dump = cls().model_dump()
    assert dump["kind"] == legacy_kind
    restored = Normalization.model_validate(dump)
    assert isinstance(restored, cls)
