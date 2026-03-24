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


class RandomSplitter(BaseModel):
    """Random train / validation / test split."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["random"] = "random"
    val_split: float = 0.2
    test_split: float = 0.0
    stratify: bool = True
    seed: int = 42

    def split(self, dataset, metadata):
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Subset

        indices = list(range(len(dataset)))
        stratify_col = metadata["target"].values if self.stratify and "target" in metadata.columns else None

        test_ds = None
        trainval_idx = indices
        if self.test_split > 0:
            trainval_idx, test_idx = train_test_split(
                indices, test_size=self.test_split, random_state=self.seed, stratify=stratify_col,
            )
            test_ds = Subset(dataset, test_idx)
            stratify_col = metadata.loc[trainval_idx, "target"].values if stratify_col is not None else None

        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=self.val_split / (1 - self.test_split) if self.test_split > 0 else self.val_split,
            random_state=self.seed, stratify=stratify_col,
        )
        return Subset(dataset, train_idx), Subset(dataset, val_idx), test_ds


class CrossSubjectSplitter(BaseModel):
    """Leave-subjects-out cross-validation split."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["cross_subject"] = "cross_subject"
    fold: int = 0
    n_folds: int = 5
    val_split: float = 0.2
    stratify: bool = True
    seed: int = 42

    def split(self, dataset, metadata):
        import numpy as np
        from moabb.evaluations.splitters import CrossSubjectSplitter as _CSS
        from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, train_test_split
        from torch.utils.data import Subset

        y = metadata["target"].values if "target" in metadata.columns else np.zeros(len(metadata))
        cv_class = StratifiedGroupKFold if self.stratify else GroupKFold
        splitter = _CSS(n_splits=self.n_folds, cv_class=cv_class)
        splits = list(splitter.split(y, metadata))

        if self.fold >= len(splits):
            raise ValueError(f"fold {self.fold} >= n_splits {len(splits)}")

        train_idx, test_idx = splits[self.fold]
        test_ds = Subset(dataset, list(test_idx))

        strat = metadata.loc[train_idx, "target"].values if self.stratify and "target" in metadata.columns else None
        tr, va = train_test_split(list(train_idx), test_size=self.val_split, random_state=self.seed, stratify=strat)
        return Subset(dataset, tr), Subset(dataset, va), test_ds


class PredefinedSplitter(BaseModel):
    """Split by predefined metadata values (e.g. subject IDs)."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["predefined"] = "predefined"
    metadata_key: str
    train_values: list[int | str]
    val_values: list[int | str]
    test_values: list[int | str] | None = None

    def split(self, dataset, metadata):
        from torch.utils.data import Subset

        col = metadata[self.metadata_key]

        def _indices(values):
            return list(metadata.index[col.isin(values)])

        train_idx = _indices(self.train_values)
        val_idx = _indices(self.val_values)
        test_ds = Subset(dataset, _indices(self.test_values)) if self.test_values else None
        return Subset(dataset, train_idx), Subset(dataset, val_idx), test_ds


Splitter = Annotated[
    Union[RandomSplitter, CrossSubjectSplitter, PredefinedSplitter],
    Field(discriminator="kind"),
]


# ============================================================================
# Dataset
# ============================================================================


class Dataset(BaseModel):
    """A downstream EEG dataset loaded from HuggingFace Hub."""

    model_config = ConfigDict(extra="forbid")

    hf_id: str = Field(description="HuggingFace Hub dataset ID, e.g. 'braindecode/bcic2a'.")
    n_classes: int | None = Field(description="Number of classes (None for regression).")
    splitter: Splitter
    batch_size: int = 64
    num_workers: int = 0

    def load(self):
        """Pull windowed dataset from HuggingFace Hub."""
        from braindecode.datasets import BaseConcatDataset

        log.info("Loading dataset from HuggingFace Hub: %s", self.hf_id)
        return BaseConcatDataset.pull_from_hub(self.hf_id)

    def setup(self, normalization=None):
        """Load dataset, apply normalization, split, and return sets.

        Returns
        -------
        tuple of (full_dataset, train, val, test, info_dict)
            info_dict has keys: n_chans, n_times, n_outputs, sfreq, chs_info.
        """
        windows = self.load()

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
        mne_info = ds0.raw.info if hasattr(ds0, "raw") and ds0.raw is not None else ds0.windows.info
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
