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
    ScratchBackbone,
    _BackboneBase,
    _BraindecodeBackbone,
)
from open_eeg_bench.dataset import Dataset
from open_eeg_bench.finetuning import AdaLoRA, Finetuning, Frozen, FullFinetune
from open_eeg_bench.head import Head, LinearHead, OriginalHead, FlattenHead
from open_eeg_bench.training import Training, RidgeProbingTraining

# Backbone union with discriminator so exca can compute deterministic UIDs.
_Backbone = Annotated[
    Union[PretrainedBackbone, ScratchBackbone, PlaceholderBackbone],
    Field(discriminator="kind"),
]

_TrainingConfig = Annotated[
    Union[Training, RidgeProbingTraining],
    Field(discriminator="kind"),
]

log = logging.getLogger(__name__)


class Experiment(BaseModel):
    """One fine-tuning experiment = backbone x head x finetuning x dataset."""

    model_config = ConfigDict(extra="forbid")

    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("verbose",)

    seed: int = 0
    backbone: _Backbone
    head: Head = Field(default_factory=LinearHead)
    finetuning: Finetuning = Field(default_factory=Frozen)
    dataset: Dataset
    training: _TrainingConfig = Field(default_factory=Training)
    infra: exca.TaskInfra = exca.TaskInfra(version="1")
    verbose: int = 1

    @model_validator(mode="after")
    def _check_consistency(self):
        is_ridge = self.training.kind == "ridge"
        is_flatten_head = isinstance(self.head, FlattenHead)

        if is_ridge and not is_flatten_head:
            raise ValueError(
                "Ridge probing requires FlattenHead (head='flatten_head')."
            )
        if is_flatten_head and not is_ridge:
            raise ValueError(
                "FlattenHead can only be used with ridge probing training."
            )

        if is_ridge and not isinstance(self.finetuning, Frozen):
            raise ValueError("Ridge probing requires Frozen finetuning.")

        if is_ridge and self.backbone.training_required_modules:
            raise ValueError(
                f"Ridge probing incompatible with backbone.training_required_modules="
                f"{self.backbone.training_required_modules}. "
                f"These backbones need SGD training."
            )

        if (
            not is_ridge
            and isinstance(self.finetuning, Frozen)
            and isinstance(self.head, OriginalHead)
        ):
            raise ValueError(
                "Frozen finetuning with OriginalHead trains nothing new. "
                "Use LinearHead or MLPHead instead."
            )

        if isinstance(self.backbone, ScratchBackbone) and not isinstance(
            self.finetuning, FullFinetune
        ):
            raise ValueError(
                "ScratchBackbone can only be used with FullFinetune finetuning "
                "(no pretrained weights to freeze or adapt)."
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
        import time
        import numpy as np
        import torch

        verbose = self.verbose

        # ===============================================================
        # 0. Seed EVERYTHING for reproducibility
        # ===============================================================
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
        except Exception as e:
            log.warning(f"Skipped CUDA determinism settings: {e}")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # ===============================================================
        # 1. Load data
        # ===============================================================
        backbone_obj: _BackboneBase = self.backbone  # type: ignore[assignment]
        full_ds, train_set, val_set, test_set, info = self.dataset.setup(
            normalization=backbone_obj.normalization
        )
        if verbose:
            print(
                f"Data: {len(train_set)} train, {len(val_set)} val, "
                f"{len(test_set)} test | "
                f"shape=({info['n_chans']}, {info['n_times']}) "
                f"sfreq={info['sfreq']:.0f}"
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
        # 2.1. Initialize lazy modules with a dummy forward pass
        # ===============================================================
        self._initialize_lazy_modules(model, info)

        # ===============================================================
        # 3. Apply head
        # ===============================================================
        self.head.apply(model, info["n_outputs"], backbone_obj.head_module_name)

        # ===============================================================
        # 3.1. Initialize lazy modules with a dummy forward pass
        # ===============================================================
        self._initialize_lazy_modules(model, info)

        # ===============================================================
        # 4. Apply finetuning
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

        model, adapter_stats = finetuning.apply(model, backbone_obj)

        if verbose:
            print(
                f"Finetuning: {adapter_stats['method']} — "
                f"{adapter_stats['trainable_params']:,}/"
                f"{adapter_stats['total_params']:,} params "
                f"({adapter_stats['trainable_pct']:.1f}%)"
            )

        # ===============================================================
        # 5. Get skorch callbacks from the finetuning
        # ===============================================================
        callbacks = self.finetuning.get_callbacks()

        # ===============================================================
        # 6. Create learner and train
        # ===============================================================
        if verbose:
            print(f"Training: {self.training.kind}")

        learner = self.training.build_learner(
            model=model,
            callbacks=callbacks,
            n_classes=self.dataset.n_classes,
            val_set=val_set,
            verbose=verbose,
            seed=self.seed,
        )
        t0 = time.time()
        learner.fit(train_set, y=None)
        fit_time = time.time() - t0

        # ===============================================================
        # 7. Test
        # ===============================================================
        results = {"adapter_stats": adapter_stats, "fit_time": fit_time}
        y_pred = learner.predict(test_set)
        y_true = np.array([test_set[i][1] for i in range(len(test_set))])
        is_regression = self.dataset.n_classes is None
        if is_regression:
            from sklearn.metrics import r2_score

            results["test_r2"] = float(r2_score(y_true, y_pred.ravel()))
        else:
            from sklearn.metrics import balanced_accuracy_score

            results["test_balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

        if verbose:
            metric_name = "test_r2" if is_regression else "test_balanced_accuracy"
            print(
                f"Test score: {metric_name} = {results[metric_name]:.4f} "
                f"(fit time: {fit_time:.1f}s)"
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


def collect_completed_results(
    experiments: Sequence[Experiment], wait: bool = False, collect_all: bool = False
) -> "pd.DataFrame":
    """Collect results of experiments.

    This function never launches any jobs; it only collects results.

    Parameters
    ----------
    experiments: Sequence[Experiment]
        The list of experiments to collect results from.
    wait: bool
        If True, wait for all experiments to reach a terminal status before collecting results.
        If False, collect results for experiments that have already reached a terminal status and skip the rest.
        However, experiments with status "not submitted" will not be launched.
    collect_all: bool
        If True, collect one row per experiment but will not wait for 'running' experiments to finish (unless wait=True).
        If False (and wait=False), only collect results for experiments with status "completed".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the collected results.
    """
    import pandas as pd

    def parse_exception(job):
        ex = str(job.exception())
        if "failed during processing with trace" in ex:
            # https://github.com/facebookincubator/submitit/blob/ca51a66b6da2400468f338133eabdfb4c9a2936c/submitit/core/core.py#L332
            return ex.split("--------------")[1].strip().splitlines()[-1]
        if "has not produced any output" in ex:
            # https://github.com/facebookincubator/submitit/blob/ca51a66b6da2400468f338133eabdfb4c9a2936c/submitit/core/core.py#L373
            return ex.splitlines()[0]
        last_line = ex.strip().splitlines()[-1]
        return last_line

    rows = []
    status_counts = {}
    for exp in experiments:
        status = exp.infra.status()
        if (status != "completed") and not (collect_all or wait):
            log.info(
                f"Experiment {exp.infra.uid()} has status '{status}', skipping result collection."
            )
            continue
        assert isinstance(  # for type checking
            (backbone := exp.backbone), _BraindecodeBackbone
        )
        row = {
            "backbone": backbone.model_cls.split(".")[-1],
            "dataset": exp.dataset.hf_id,
            'training': exp.training.kind,
            "finetuning": exp.finetuning.kind,
            "head": exp.head.kind,
            "seed": exp.seed,
        }
        if status != "not submitted":
            job = exp.infra.job()
            row["job_id"] = job.job_id
            if wait:
                job.wait()
                status = exp.infra.status()  # refresh status after waiting
            if status == "failed":
                row["exception"] = parse_exception(job)
            if status == "completed":
                result = job.result()
                row.update(result)
        status_counts[status] = status_counts.get(status, 0) + 1
        row["status"] = status
        rows.append(row)

    # Print summary of job statuses:
    log.info(f"Experiment status summary: {status_counts}")
    log.info(
        f"Returning results for {len(rows)}/{len(experiments)} completed experiments."
    )

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def run_many(
    experiments: Sequence[Experiment],
    max_workers: int = 256,
    wait: bool = False,
    collect_all: bool = False,
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
    wait: bool
        If True, wait for all experiments to reach a terminal status before collecting results.
        If False, collect results for experiments that have already reached a terminal status and skip the rest.
        However, experiments with status "not submitted" will not be launched.
    collect_all: bool
        If True, collect one row per experiment but will not wait for 'running' experiments to finish (unless wait=True).
        If False (and wait=False), only collect results for experiments with status "completed".

    Returns
    -------
    pd.DataFrame
        One row per experiment with flattened result columns.
    """
    if any(exp.infra.folder is None for exp in experiments):
        raise ValueError("Each experiment must have infra.folder set for caching")

    if not experiments:
        import pandas as pd

        return pd.DataFrame()

    first = experiments[0]

    # Launch the jobs (non-blocking for slurm/local, blocking for cluster=None)
    with first.infra.job_array(max_workers=max_workers) as array:
        array.extend(experiments)

    return collect_completed_results(experiments, wait=wait, collect_all=collect_all)
