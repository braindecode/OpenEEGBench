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
    def _check_frozen_needs_new_head(self):
        if isinstance(self.finetuning, Frozen) and isinstance(self.head, OriginalHead):
            raise ValueError(
                "Frozen finetuning with OriginalHead trains nothing new. "
                "Use LinearHead or MLPHead instead."
            )
        return self

    @infra.apply()
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

        # ===============================================================
        # 0. Seed EVERYTHING for reproducibility
        # ===============================================================
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Handle CUDA determinism carefully
        try:
            import multiprocessing

            # Check if we are in a worker process
            is_worker = (
                "LokyProcess" in multiprocessing.current_process().name
                or "SpawnPoolWorker" in multiprocessing.current_process().name
            )
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
            total_step = (
                max(1, len(train_set) // self.training.batch_size)
                * self.training.max_epochs
            )
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
        # 6. Get skorch callbacks from the finetuning
        # ===============================================================
        callbacks = self.finetuning.get_callbacks()

        # ===============================================================
        # 7. Create learner and train
        # ===============================================================
        learner = self.training.build_learner(
            model=model,
            callbacks=callbacks,
            n_classes=self.dataset.n_classes,
            val_set=val_set,
        )
        learner.fit(train_set, y=None)

        # ===============================================================
        # 8. Test
        # ===============================================================
        results = {"adapter_stats": adapter_stats}
        y_pred = learner.predict(test_set)
        y_true = np.array([test_set[i][1] for i in range(len(test_set))])
        is_regression = self.dataset.n_classes is None
        if is_regression:
            from sklearn.metrics import r2_score

            results["test_r2"] = float(r2_score(y_true, y_pred.ravel()))
            log.info("Test R²: %.4f", results["test_r2"])
        else:
            from sklearn.metrics import balanced_accuracy_score

            results["test_balanced_accuracy"] = balanced_accuracy_score(
                y_true, y_pred
            )
            log.info(
                "Test balanced accuracy: %.4f", results["test_balanced_accuracy"]
            )

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


def run_many(
    experiments: Sequence[Experiment],
    max_workers: int = 256,
) -> "pd.DataFrame":
    """Run a list of experiments with caching and optional cluster submission.

    Execution mode is controlled by the ``infra`` field on each experiment:

    * ``cluster=None`` -- run sequentially in the current process.
    * ``cluster="local"`` -- run in parallel as local subprocesses
      (all jobs are launched at once; ``max_workers`` is ignored).
    * ``cluster="slurm"`` -- submit as a SLURM job array with at most
      ``max_workers`` jobs running simultaneously.

    All experiments must share the same ``infra.folder`` and ``infra.cluster``
    settings.  Per-experiment overrides (seed, dataset, etc.) are fine.

    Parameters
    ----------
    experiments : sequence of Experiment
        Experiments to run.  Each must have ``infra.folder`` set for caching.
    max_workers : int
        Maximum number of SLURM jobs running at the same time (maps to
        ``--array=...%max_workers``).  Only effective with ``cluster="slurm"``.

    Returns
    -------
    pd.DataFrame
        One row per experiment with flattened result columns.
    """
    import pandas as pd

    if not experiments:
        return pd.DataFrame()

    first = experiments[0]

    # Launch the jobs (non-blocking for slurm/local, blocking for cluster=None)
    with first.infra.job_array(max_workers=max_workers) as array:
        array.extend(experiments)

    # Collect results of the completed jobs
    rows = []
    status_counts = {}
    for exp in experiments:
        status = exp.infra.status()
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "completed":
            result = exp.run()
            result["dataset"] = exp.dataset.hf_id
            result["finetuning"] = exp.finetuning.kind
            result["head"] = exp.head.kind
            result["seed"] = exp.seed
            rows.append(result)
        else:
            log.info(
                f"Experiment {exp.infra.uid()} has status '{status}', skipping result collection."
            )
    # Print summary of job statuses:
    log.info(f"Experiment status summary: {status_counts}")
    log.info(f"Returning results for {len(rows)}/{len(experiments)} completed experiments.")

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()
