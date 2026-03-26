"""Post-window normalization transforms for EEG data.

Each EEG foundation model was pretrained with a specific normalization.
These config classes reproduce that normalization so fine-tuning data
matches the pretraining distribution.

All normalizations operate on a single window (channels x time) as a
numpy array and are applied as a transform after windowing.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DivideByConstant(BaseModel):
    """Divide data by a constant factor.

    Used by LaBraM and CBraMod (factor=100) to set the EEG unit to 0.1 mV.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["divide_by_constant"] = "divide_by_constant"
    factor: float = 100.0

    def apply(self, data: np.ndarray) -> np.ndarray:
        return data / self.factor


class PercentileScale(BaseModel):
    """Per-channel percentile normalization.

    Used by BIOT: each channel is divided by the q-th percentile of its
    absolute amplitude so that the bulk of values falls near [-1, 1].
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["percentile_scale"] = "percentile_scale"
    q: float = 95.0
    eps: float = 1e-8

    def apply(self, data: np.ndarray) -> np.ndarray:
        quantile = np.quantile(
            np.abs(data), q=self.q / 100.0, axis=-1, keepdims=True, method="linear"
        )
        return data / (quantile + self.eps)


class MinMaxScale(BaseModel):
    """Per-window min-max scaling to [-1, 1].

    Used by BENDR.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["minmax_scale"] = "minmax_scale"

    def apply(self, data: np.ndarray) -> np.ndarray:
        dmin = np.min(data)
        dmax = np.max(data)
        drange = dmax - dmin
        if drange < 1e-10:
            return np.zeros_like(data)
        return 2.0 * (data - dmin) / drange - 1.0


class WindowZScore(BaseModel):
    """Per-window z-score normalization with optional sigma clipping.

    Used by REVE (clip_sigma=15) and EEGPT (channel_wise=True).
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["window_zscore"] = "window_zscore"
    channel_wise: bool = False
    clip_sigma: float | None = 15.0
    eps: float = 1e-10

    def apply(self, data: np.ndarray) -> np.ndarray:
        axis = -1 if self.channel_wise else None
        mean = np.mean(data, keepdims=True, axis=axis)
        std = np.std(data, keepdims=True, axis=axis)
        std = np.maximum(std, self.eps)
        normalised = (data - mean) / std
        if self.clip_sigma is not None:
            normalised = np.clip(normalised, -self.clip_sigma, self.clip_sigma)
        return normalised


class ScaleToMV(BaseModel):
    """Convert microvolt data to millivolts (divide by 1000).

    Used by EEGPT during pretraining.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["scale_to_mv"] = "scale_to_mv"

    def apply(self, data: np.ndarray) -> np.ndarray:
        return data / 1000.0


Normalization = Annotated[
    Union[DivideByConstant, PercentileScale, MinMaxScale, WindowZScore, ScaleToMV],
    Field(discriminator="kind"),
]
