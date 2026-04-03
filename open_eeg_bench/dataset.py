"""Dataset and splitting configurations.

Datasets are loaded from HuggingFace Hub as pre-windowed BaseConcatDataset
objects.  Splitting strategies define how to partition into train/val/test.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from torch.utils.data import Subset
    from braindecode.datasets import BaseConcatDataset

log = logging.getLogger(__name__)


# ============================================================================
# Splitters
# ============================================================================


class PredefinedSplitter(BaseModel):
    """Split by predefined metadata values (e.g. subject IDs).

    Validation set is specified either by explicit ``val_values`` or by
    randomly sampling ``val_split`` fraction from the training indices.
    Exactly one of the two must be provided.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["predefined"] = "predefined"
    metadata_key: str
    train_values: list[int | str]
    val_values: list[int | str] | None = None
    val_size: float | None = None
    test_values: list[int | str]

    @model_validator(mode="after")
    def check_val_source(self):
        if (self.val_values is not None) == (self.val_size is not None):
            raise ValueError("Exactly one of val_values or val_size must be provided.")
        return self

    def split(self, dataset, metadata):
        from torch.utils.data import Subset

        col = metadata[self.metadata_key]

        def _indices(values):
            return list(metadata.index[col.isin(values)])

        all_train_idx = _indices(self.train_values)
        test_idx = _indices(self.test_values)

        if self.val_values is not None:
            train_idx = all_train_idx
            val_idx = _indices(self.val_values)
        else:
            from sklearn.model_selection import GroupShuffleSplit

            subjects = metadata["subject"].values
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.val_size,
                random_state=12,
            )
            train_idx, val_idx = next(
                gss.split(all_train_idx, groups=subjects[all_train_idx])
            )

        return (
            Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx),
        )


Splitter = Annotated[
    Union[PredefinedSplitter],
    Field(discriminator="kind"),
]


# ============================================================================
# Dataset
# ============================================================================


class Dataset(BaseModel):
    """A downstream EEG dataset loaded from HuggingFace Hub."""

    model_config = ConfigDict(extra="forbid")

    hf_id: str = Field(
        description="HuggingFace Hub dataset ID, e.g. 'braindecode/bcic2a'."
    )
    n_classes: int | None = Field(
        description="Number of classes (None for regression)."
    )
    splitter: PredefinedSplitter
    montage_name: str | None = Field(
        default=None,
        description=(
            "Name of an MNE standard montage to set on the data, "
            "e.g. 'standard_1005'. Use when channel positions are missing."
        ),
    )

    def load(self):
        """Pull windowed dataset from HuggingFace Hub."""
        from braindecode.datasets import BaseConcatDataset

        log.info("Loading dataset from HuggingFace Hub: %s", self.hf_id)
        return BaseConcatDataset.pull_from_hub(self.hf_id, preload=False)

    def setup(self, normalization=None):
        """Load dataset, apply normalization, split, and return sets.

        Returns
        -------
        tuple of (full_dataset, train, val, test, info_dict)
            info_dict has keys: n_chans, n_times, n_outputs, sfreq, chs_info.
        """
        windows = self.load()

        # Set standard montage if channel positions are missing
        if self.montage_name is not None:
            import mne

            montage = mne.channels.make_standard_montage(self.montage_name)
            for ds in windows.datasets:
                if hasattr(ds, "raw") and ds.raw is not None:
                    ds.raw.set_montage(montage)
                else:
                    ds.windows.set_montage(montage)

        # Apply normalization as transform
        if normalization is not None:
            for ds in windows.datasets:
                existing = ds.transform

                def _make_norm_transform(norm, old_transform):
                    def transform(x):
                        x = norm.apply(x)
                        if old_transform is not None:
                            x = old_transform(x)
                        return x

                    return transform

                ds.transform = _make_norm_transform(normalization, existing)

        # Extract dataset info (works for both raw-backed and Epochs-backed datasets)
        sample_x, sample_y, _ = windows[0]
        n_times = sample_x.shape[-1]
        ds0 = windows.datasets[0]
        mne_info = (
            ds0.raw.info
            if hasattr(ds0, "raw") and ds0.raw is not None
            else ds0.windows.info
        )
        sfreq = mne_info["sfreq"]
        chs_info = mne_info["chs"]
        n_chans = len(chs_info)
        n_outputs = self.n_classes if self.n_classes is not None else 1

        # Split
        metadata = windows.get_metadata().reset_index(drop=True)
        train_set, val_set, test_set = self.splitter.split(windows, metadata)

        info = dict(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            sfreq=sfreq,
            chs_info=chs_info,
        )
        return windows, train_set, val_set, test_set, info
