"""Test normalization transforms."""

import numpy as np
import pytest

from open_eeg_bench.normalization import (
    DivideByConstant,
    MinMaxScale,
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
