"""Training configuration.

Maps to skorch / braindecode EEGClassifier parameters: optimizer,
scheduler, early stopping, checkpointing, and logging.
"""

from __future__ import annotations

from typing import ClassVar, Literal, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from skorch.callbacks import Callback


class _BaseCallbackConfig(BaseModel):
    """Base for all callback configs."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False

    def _build_callback(self) -> "Callback":
        raise NotImplementedError

    def build_callbacks(self) -> list["Callback"]:
        """Build skorch callbacks from config."""
        if not self.enabled:
            return []
        return [self._build_callback()]


class EarlyStopping(_BaseCallbackConfig):
    enabled: bool = True
    patience: int = 10
    monitor: str = "valid_loss"
    lower_is_better: bool = True

    def _build_callback(self):
        from skorch.callbacks import EarlyStopping as SkorchEarlyStopping

        return SkorchEarlyStopping(
            monitor=self.monitor,
            patience=self.patience,
            lower_is_better=self.lower_is_better,
        )


class Checkpoint(_BaseCallbackConfig):
    monitor: str = "valid_acc"
    lower_is_better: bool = False

    def _build_callback(self):
        from skorch.callbacks import Checkpoint

        return Checkpoint(
            monitor=self.monitor,
            f_params="best_model.pt",
            lower_is_better=self.lower_is_better,
        )


class WandbConfig(_BaseCallbackConfig):
    project: str = "open-eeg-bench"
    entity: str | None = None

    def _build_callback(self):
        from skorch.callbacks import WandbLogger
        import wandb

        wandb_run = wandb.init(project=self.project, entity=self.entity)
        return WandbLogger(wandb_run=wandb_run, save_model=False)


class Training(BaseModel):
    """Training hyper-parameters and callback settings."""

    model_config = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("device", "num_workers")
    kind: Literal["sgd"] = "sgd"

    max_epochs: int = 50
    lr: float = 5e-4
    gradient_clip_val: float | None = 1.0
    batch_size: int = 64
    num_workers: int = 0
    device: str = "cpu"

    # Scheduler
    use_scheduler: bool = True
    scheduler_warmup_epochs: int = 5
    scheduler_eta_min: float = 1e-6

    # Callbacks
    early_stopping: EarlyStopping = Field(default_factory=EarlyStopping)
    checkpoint: Checkpoint = Field(default_factory=Checkpoint)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    def build_callbacks(self, is_regression) -> list:
        """Build skorch callbacks from training config."""
        from skorch.callbacks import EpochScoring, GradientNormClipping, LRScheduler

        cbs = []

        # Scoring
        scoring, lower_is_better = (
            ("r2", True) if is_regression else ("balanced_accuracy", False)
        )
        cbs.append(
            EpochScoring(
                scoring,
                name="valid_score",
                lower_is_better=lower_is_better,
                on_train=False,
            )
        )
        cbs.append(
            EpochScoring(
                scoring,
                name="train_score",
                lower_is_better=lower_is_better,
                on_train=True,
            )
        )

        # Early stopping
        cbs.extend(self.early_stopping.build_callbacks())
        # Checkpointing
        cbs.extend(self.checkpoint.build_callbacks())
        # Logging
        cbs.extend(self.wandb.build_callbacks())

        # Gradient clipping
        if self.gradient_clip_val is not None:
            cbs.append(GradientNormClipping(gradient_clip_value=self.gradient_clip_val))

        # LR scheduler
        if self.use_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR

            cbs.append(
                LRScheduler(
                    policy=CosineAnnealingLR,
                    T_max=self.max_epochs,
                    eta_min=self.scheduler_eta_min,
                )
            )

        return cbs

    def build_learner(self, model, callbacks, n_classes, val_set):
        from torch.optim import AdamW
        from skorch.helper import predefined_split
        from braindecode import EEGClassifier, EEGRegressor

        is_regression = n_classes is None
        callbacks = list(callbacks)
        callbacks.extend(self.build_callbacks(is_regression))

        common_kwargs = dict(
            module=model,
            optimizer=AdamW,
            optimizer__lr=self.lr,
            train_split=predefined_split(val_set),
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            device=self.device,
            callbacks=callbacks,
            verbose=1,
            iterator_train__num_workers=self.num_workers,
            iterator_valid__num_workers=self.num_workers,
        )

        is_regression = n_classes is None
        if is_regression:
            learner = EEGRegressor(**common_kwargs)
        else:
            classes = list(range(n_classes))
            learner = EEGClassifier(classes=classes, **common_kwargs)

        return learner


class RidgeProbingTraining(BaseModel):
    """Closed-form ridge regression probing on frozen backbone features.

    Runs three streaming passes: accumulate sufficient statistics on train,
    select regularization strength on val, predict on test. No gradient-based
    training, no callbacks.
    """

    model_config = ConfigDict(extra="forbid")
    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("device", "num_workers")

    kind: Literal["ridge"] = "ridge"
    batch_size: int = 64
    num_workers: int = 0
    device: str = "cpu"
    lambdas: list[float] | None = None  # None → default logspace × eigval scale

    def build_learner(self, model, callbacks, n_classes, val_set):
        from open_eeg_bench.ridge_probe import StreamingRidgeProbeLearner

        # callbacks ignored — no epochs, no early stopping
        return StreamingRidgeProbeLearner(
            feature_extractor=model,
            n_classes=n_classes,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            lambdas=self.lambdas,
            val_set=val_set,
        )
