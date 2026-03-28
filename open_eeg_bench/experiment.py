"""Top-level Experiment configuration and runner.

An ``Experiment`` instance fully describes a single fine-tuning run:
one backbone, one head, one fine-tuning strategy, one dataset split.
Calling ``experiment.run()`` executes the full pipeline and returns
the test metric.

The ``run_many()`` module-level function runs a list of experiments
with caching.  On SLURM it submits them as a job array; locally it
runs them sequentially or in a process pool depending on the ``cluster``
setting of the first experiment's ``infra``.
"""

import logging
from typing import Annotated, ClassVar, Sequence, Union, TYPE_CHECKING

import exca
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import pandas as pd

from open_eeg_bench.backbone import (
    PlaceholderBackbone,
    PretrainedBackbone,
    _BackboneBase,
)
from open_eeg_bench.dataset import Dataset
from open_eeg_bench.finetuning import AdaLoRA, Finetuning, Frozen
from open_eeg_bench.head import Head, LinearHead, OriginalHead
from open_eeg_bench.training import Training

# Backbone union with discriminator so exca can compute deterministic UIDs.
_Backbone = Annotated[
    Union[PretrainedBackbone, PlaceholderBackbone],
    Field(discriminator="kind"),
]

log = logging.getLogger(__name__)


class Experiment(BaseModel):
    """One fine-tuning experiment = backbone x head x finetuning x dataset."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    backbone: _Backbone
    head: Head = Field(default_factory=LinearHead)
    finetuning: Finetuning = Field(default_factory=Frozen)
    dataset: Dataset
    training: Training = Field(default_factory=Training)
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("infra",)

    @model_validator(mode="after")
    def _check_not_placeholder(self):
        if isinstance(self.backbone, PlaceholderBackbone):
            raise ValueError(
                "PlaceholderBackbone cannot be used to run an experiment. "
                "Replace it with a concrete backbone (e.g. PretrainedBackbone)."
            )
        return self

    @model_validator(mode="after")
    def _check_frozen_needs_new_head(self):
        if isinstance(self.finetuning, Frozen) and isinstance(self.head, OriginalHead):
            raise ValueError(
                "Frozen finetuning with OriginalHead trains nothing new. "
                "Use LinearHead or MLPHead instead."
            )
        return self

    @infra.apply(
        exclude_from_cache_uid=(
            "training.device",
            "training.batch_size",
            "dataset.batch_size",
            "dataset.num_workers",
        ),
    )
    def run(self) -> dict:
        """Execute the full training pipeline.

        Returns
        -------
        dict
            Results including test metrics and adapter stats.
        """
        import os
        import numpy as np
        import torch
        import torch.nn as nn

        # ===============================================================
        # 0. Seed EVERYTHING for reproducibility
        # ===============================================================
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Handle CUDA determinism carefully
        try:
            import multiprocessing
            # Check if we are in a worker process
            is_worker = "LokyProcess" in multiprocessing.current_process().name or "SpawnPoolWorker" in multiprocessing.current_process().name
            # Use environment heuristic or process name
            if torch.cuda.is_available():
                 # Only set CUDA seeds if we are in a worker or explicit single-run
                 torch.cuda.manual_seed(self.seed)
                 torch.cuda.manual_seed_all(self.seed)

                 if hasattr(torch.backends, "cudnn"):
                      torch.backends.cudnn.benchmark = False
                      torch.backends.cudnn.deterministic = True
        except Exception as e:
            log.warning(f"Skipped CUDA determinism settings: {e}")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        log.info(f"Set random seed to {self.seed} with full determinism enabled")

        # ===============================================================
        # 1. Load data
        # ===============================================================
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

        # ===============================================================
        # 2. Build model (and load pretrained weights)
        # ===============================================================
        model = backbone_obj.build(
            n_chans=info["n_chans"],
            n_times=info["n_times"],
            n_outputs=info["n_outputs"],
            sfreq=info["sfreq"],
            chs_info=info["chs_info"],
        )

        # ===============================================================
        # 3. Apply head
        # ===============================================================
        self.head.apply(model, info["n_outputs"], backbone_obj.head_module_name)

        # ===============================================================
        # 4. Initialize lazy modules with a dummy forward pass
        # ===============================================================
        self._initialize_lazy_modules(model, info)

        # ===============================================================
        # 5. Apply finetuning
        # ===============================================================
        finetuning = self.finetuning
        if isinstance(finetuning, AdaLoRA) and finetuning.total_step is None:
            batch_size = self.training.batch_size or self.dataset.batch_size
            total_step = max(1, len(train_set) // batch_size) * self.training.max_epochs
            updates = {"total_step": total_step}
            # Ensure tinit + tfinal < total_step for the budgeting phase
            if finetuning.tinit + finetuning.tfinal >= total_step:
                updates["tinit"] = total_step // 5
                updates["tfinal"] = total_step // 2
            finetuning = finetuning.model_copy(update=updates)
            log.info("AdaLoRA: auto-computed total_step=%d", total_step)
        model, adapter_stats = finetuning.apply(model, backbone_obj)
        log.info(
            "Finetuning: %s — %s/%s params (%.1f%%)",
            adapter_stats["method"],
            f"{adapter_stats['trainable_params']:,}",
            f"{adapter_stats['total_params']:,}",
            adapter_stats["trainable_pct"],
        )

        # ===============================================================
        # 6. Build skorch callbacks
        # ===============================================================
        is_regression = self.dataset.n_classes is None
        callbacks = self._build_callbacks(regression=is_regression)
        callbacks.extend(self.finetuning.get_callbacks())

        # ===============================================================
        # 7. Create learner and train
        # ===============================================================
        batch_size = self.training.batch_size or self.dataset.batch_size
        common_kwargs = dict(
            module=model,
            optimizer=torch.optim.AdamW,
            optimizer__lr=self.training.lr,
            optimizer__weight_decay=self.training.weight_decay,
            max_epochs=self.training.max_epochs,
            batch_size=batch_size,
            device=self.training.device,
            callbacks=callbacks,
            train_split=None,
            verbose=1,
            iterator_train__num_workers=self.dataset.num_workers,
            iterator_valid__num_workers=self.dataset.num_workers,
        )

        if is_regression:
            from braindecode import EEGRegressor
            learner = EEGRegressor(criterion=nn.MSELoss, **common_kwargs)
        else:
            from braindecode import EEGClassifier
            classes = list(range(info["n_outputs"])) if self.dataset.n_classes else None
            learner = EEGClassifier(
                criterion=nn.CrossEntropyLoss, classes=classes, **common_kwargs,
            )

        from skorch.helper import predefined_split
        learner.train_split = predefined_split(val_set)
        learner.fit(train_set, y=None)

        # ===============================================================
        # 8. Test
        # ===============================================================
        results = {"adapter_stats": adapter_stats}
        if test_set is not None:
            y_pred = learner.predict(test_set)
            y_true = np.array([test_set[i][1] for i in range(len(test_set))])
            if is_regression:
                from sklearn.metrics import r2_score
                results["test_r2"] = float(r2_score(y_true, y_pred.ravel()))
                log.info("Test R²: %.4f", results["test_r2"])
            else:
                from sklearn.metrics import balanced_accuracy_score
                results["test_balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
                log.info("Test balanced accuracy: %.4f", results["test_balanced_accuracy"])
        else:
            log.info("No test set configured.")

        return results

    @staticmethod
    def _initialize_lazy_modules(model, info: dict) -> None:
        """Run a dummy forward pass to materialize LazyLinear modules."""
        import torch
        import torch.nn as nn

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

    def _build_callbacks(self, regression: bool = False) -> list:
        """Build skorch callbacks from training config."""
        from skorch.callbacks import (
            EarlyStopping as SkorchEarlyStopping,
            EpochScoring,
            GradientNormClipping,
            LRScheduler,
        )

        cbs = []

        # Scoring
        scoring = "r2" if regression else "balanced_accuracy"
        cbs.append(
            EpochScoring(scoring, name="valid_score", lower_is_better=False, on_train=False)
        )
        cbs.append(
            EpochScoring(scoring, name="train_score", lower_is_better=False, on_train=True)
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
            import torch
            cbs.append(
                LRScheduler(
                    policy=torch.optim.lr_scheduler.CosineAnnealingLR,
                    T_max=self.training.max_epochs,
                    eta_min=self.training.scheduler_eta_min,
                )
            )

        return cbs


def run_many(experiments: Sequence[Experiment]) -> "pd.DataFrame":
    """Run a list of experiments with caching and optional cluster submission.

    Execution mode is controlled by the ``infra`` field on each experiment:

    * ``cluster=None`` or ``cluster="local"`` -- run sequentially in-process.
    * ``cluster="slurm"`` or ``cluster="auto"`` -- submit as a SLURM job array.
    * ``cluster="processpool"`` -- run in parallel via a local process pool.

    All experiments must share the same ``infra.folder`` and ``infra.cluster``
    settings.  Per-experiment overrides (seed, dataset, etc.) are fine.

    Parameters
    ----------
    experiments : sequence of Experiment
        Experiments to run.  Each must have ``infra.folder`` set for caching.

    Returns
    -------
    pd.DataFrame
        One row per experiment with flattened result columns.
    """
    import pandas as pd

    if not experiments:
        return pd.DataFrame()

    first = experiments[0]

    # Use job_array for cluster submission (slurm, auto, processpool, etc.)
    if first.infra.cluster is not None:
        with first.infra.job_array(allow_repeated_tasks=False) as array:
            for exp in experiments:
                array.append(exp)
        # Collect results from cache for completed jobs.
        rows = []
        for exp in experiments:
            status = exp.infra.status()
            if status == "completed":
                result = exp.run()
                rows.append(result)
            else:
                log.info(
                    "Experiment %s has status '%s', skipping result collection.",
                    exp.infra.uid(),
                    status,
                )
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame()

    # Local sequential execution (cluster=None).
    rows = []
    for exp in experiments:
        try:
            result = exp.run()
            rows.append(result)
        except Exception as e:
            log.error("Experiment failed: %s", e, exc_info=True)
            rows.append({"error": str(e)})
    return pd.DataFrame(rows)
