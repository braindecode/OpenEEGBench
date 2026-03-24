"""Top-level Experiment configuration and runner.

An ``Experiment`` instance fully describes a single fine-tuning run:
one backbone, one head, one fine-tuning strategy, one dataset split.
Calling ``experiment.run()`` executes the full pipeline and returns
the test metric.

``ExperimentHandler`` wraps multiple experiments with exca ``MapInfra``
for cached, optionally parallel (SLURM) execution.
"""

import logging
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from exca import ConfDict, MapInfra
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import balanced_accuracy_score

from open_eeg_bench.backbone import Backbone, _BackboneBase
from open_eeg_bench.dataset import Dataset
from open_eeg_bench.finetuning import Finetuning, Frozen
from open_eeg_bench.head import Head, LinearHead, OriginalHead
from open_eeg_bench.training import Training

log = logging.getLogger(__name__)


class Experiment(BaseModel):
    """One fine-tuning experiment = backbone × head × finetuning × dataset."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    backbone: Backbone
    head: Head = Field(default_factory=LinearHead)
    finetuning: Finetuning = Field(default_factory=Frozen)
    dataset: Dataset
    training: Training = Field(default_factory=Training)

    @model_validator(mode="after")
    def _check_frozen_needs_new_head(self):
        if isinstance(self.finetuning, Frozen) and isinstance(self.head, OriginalHead):
            raise ValueError(
                "Frozen finetuning with OriginalHead trains nothing new. "
                "Use LinearHead or MLPHead instead."
            )
        return self

    def run(self) -> dict:
        """Execute the full training pipeline.

        Returns
        -------
        dict
            Results including test metrics and adapter stats.
        """
        # 0. Seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # 1. Load data
        backbone_obj: _BackboneBase = self.backbone  # type: ignore[assignment]
        full_ds, train_set, val_set, test_set, info = self.dataset.setup(
            normalization=backbone_obj.normalization
        )
        log.info(
            "Data: %d train, %d val, %s test | shape=(%d, %d) sfreq=%.0f",
            len(train_set),
            len(val_set),
            len(test_set) if test_set else "no",
            info["n_chans"],
            info["n_times"],
            info["sfreq"],
        )

        # 2. Build model
        model = backbone_obj.build(
            n_chans=info["n_chans"],
            n_times=info["n_times"],
            n_outputs=info["n_outputs"],
            sfreq=info["sfreq"],
            chs_info=info["chs_info"],
        )

        # 3. Load pretrained weights
        backbone_obj.load_pretrained(model)

        # 4. Apply head
        self.head.apply(model, info["n_outputs"], backbone_obj.head_module_name)

        # 5. Initialize lazy modules with a dummy forward pass
        self._initialize_lazy_modules(model, info)

        # 6. Apply finetuning
        model, adapter_stats = self.finetuning.apply(model, backbone_obj)
        log.info(
            "Finetuning: %s — %s/%s params (%.1f%%)",
            adapter_stats["method"],
            f"{adapter_stats['trainable_params']:,}",
            f"{adapter_stats['total_params']:,}",
            adapter_stats["trainable_pct"],
        )

        # 7. Build skorch callbacks
        callbacks = self._build_callbacks()
        callbacks.extend(self.finetuning.get_callbacks())

        # 8. Create EEGClassifier and train
        from braindecode import EEGClassifier

        batch_size = self.training.batch_size or self.dataset.batch_size
        classes = list(range(info["n_outputs"])) if self.dataset.n_classes else None
        clf = EEGClassifier(
            module=model,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=self.training.lr,
            optimizer__weight_decay=self.training.weight_decay,
            max_epochs=self.training.max_epochs,
            batch_size=batch_size,
            device=self.training.device,
            callbacks=callbacks,
            classes=classes,
            train_split=None,  # We handle splitting ourselves
            verbose=1,
            iterator_train__num_workers=self.dataset.num_workers,
            iterator_valid__num_workers=self.dataset.num_workers,
        )

        # Fit on train, validate on val
        from skorch.helper import predefined_split
        clf.train_split = predefined_split(val_set)
        clf.fit(train_set, y=None)

        # 9. Test
        results = {"adapter_stats": adapter_stats}
        if test_set is not None:
            y_pred = clf.predict(test_set)
            y_true = np.array([test_set[i][1] for i in range(len(test_set))])
            test_acc = balanced_accuracy_score(y_true, y_pred)
            results["test_balanced_accuracy"] = test_acc
            log.info("Test balanced accuracy: %.4f", test_acc)
        else:
            log.info("No test set configured.")

        return results

    def _initialize_lazy_modules(self, model: nn.Module, info: dict) -> None:
        """Run a dummy forward pass to materialize LazyLinear modules."""
        has_lazy = any(
            isinstance(p, nn.parameter.UninitializedParameter)
            for p in model.parameters()
        )
        if not has_lazy:
            return
        dummy = torch.zeros(1, info["n_chans"], info["n_times"])
        with torch.no_grad():
            model.eval()
            model(dummy)
            model.train()
        log.info("Initialized lazy modules with dummy forward pass")

    def _build_callbacks(self) -> list:
        """Build skorch callbacks from training config."""
        from skorch.callbacks import (
            EarlyStopping as SkorchEarlyStopping,
            EpochScoring,
            GradientNormClipping,
            LRScheduler,
        )

        cbs = []

        # Accuracy scoring
        cbs.append(
            EpochScoring(
                "balanced_accuracy",
                name="valid_acc",
                lower_is_better=False,
                on_train=False,
            )
        )
        cbs.append(
            EpochScoring(
                "balanced_accuracy",
                name="train_acc",
                lower_is_better=False,
                on_train=True,
            )
        )

        # Early stopping
        es = self.training.early_stopping
        if es.enabled:
            cbs.append(
                SkorchEarlyStopping(
                    monitor=es.monitor,
                    patience=es.patience,
                    lower_is_better=es.lower_is_better,
                )
            )

        # Gradient clipping
        if self.training.gradient_clip_val is not None:
            cbs.append(
                GradientNormClipping(
                    gradient_clip_value=self.training.gradient_clip_val
                )
            )

        # LR scheduler
        if self.training.use_scheduler:
            cbs.append(
                LRScheduler(
                    policy=torch.optim.lr_scheduler.CosineAnnealingLR,
                    T_max=self.training.max_epochs,
                    eta_min=self.training.scheduler_eta_min,
                )
            )

        return cbs


def _experiment_uid(experiment: Experiment) -> str:
    """Compute a deterministic UID for an Experiment config."""
    return ConfDict.from_model(experiment, uid=True).to_uid()


def _run_experiment(experiment: Experiment) -> dict:
    """Run a single experiment. Standalone function for use with ThreadPoolExecutor."""
    return experiment.run()


class ExperimentHandler(BaseModel):
    """Run multiple experiments with caching and optional SLURM parallelism.

    Uses exca ``MapInfra`` to cache results per experiment and optionally
    distribute runs across SLURM jobs.

    When ``parallelise_within_node=True``, experiments assigned to the same
    node are run concurrently via threads.  This is useful when individual
    runs are light (small batch size) and do not saturate the GPU alone.
    Use ``infra.min_samples_per_job`` to control how many experiments land
    on the same node.

    Examples
    --------
    Run locally, sequentially, with caching::

        handler = ExperimentHandler(infra=MapInfra(folder="./results"))
        results = handler.run([exp1, exp2, exp3])

    Run locally, 4 experiments in parallel on the same GPU::

        handler = ExperimentHandler(
            parallelise_within_node=True,
            infra=MapInfra(folder="./results", min_samples_per_job=4),
        )
        results = handler.run([exp1, exp2, exp3, exp4])

    Run on SLURM, 4 experiments per node in parallel::

        handler = ExperimentHandler(
            parallelise_within_node=True,
            infra=MapInfra(
                folder="./results",
                cluster="slurm",
                gpus_per_node=1,
                mem_gb=32,
                timeout_min=60,
                min_samples_per_job=4,
            ),
        )
        results = handler.run(experiments)
    """

    model_config = ConfigDict(extra="forbid")
    infra: MapInfra = MapInfra()
    parallelise_within_node: bool = Field(
        default=False,
        description=(
            "If True, experiments assigned to the same node are run "
            "concurrently via threads instead of sequentially."
        ),
    )

    @infra.apply(
        item_uid=_experiment_uid,
        cache_type="PandasDataFrame",
    )
    def run(
        self, experiments: Iterable[Experiment],
    ) -> Iterator[pd.DataFrame]:  # noqa: F821
        """Run experiments, yielding one DataFrame row per experiment.

        Cached experiments are skipped automatically by MapInfra.

        Parameters
        ----------
        experiments : Iterable[Experiment]
            Experiments to run.

        Yields
        ------
        pd.DataFrame
            Single-row DataFrame with flattened results for each experiment.
        """
        experiments = list(experiments)

        if self.parallelise_within_node and len(experiments) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(experiments)) as pool:
                raw_results = list(pool.map(_run_experiment, experiments))
        else:
            raw_results = [exp.run() for exp in experiments]

        for raw in raw_results:
            flat = ConfDict(raw).flat()
            yield pd.DataFrame([flat])
