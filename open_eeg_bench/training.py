"""Training configuration.

Maps to skorch / braindecode EEGClassifier parameters: optimizer,
scheduler, early stopping, checkpointing, and logging.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EarlyStopping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    patience: int = 10
    monitor: str = "valid_loss"
    lower_is_better: bool = True


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    monitor: str = "valid_acc"
    lower_is_better: bool = False


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    project: str = "open-eeg-bench"
    entity: str | None = None


class Training(BaseModel):
    """Training hyper-parameters and callback settings."""

    model_config = ConfigDict(extra="forbid")

    max_epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 0.01
    batch_size: int | None = None  # None = use dataset.batch_size
    device: str = "cpu"
    gradient_clip_val: float | None = 1.0

    # Scheduler
    use_scheduler: bool = True
    scheduler_warmup_epochs: int = 5
    scheduler_eta_min: float = 1e-6

    # Callbacks
    early_stopping: EarlyStopping = Field(default_factory=EarlyStopping)
    checkpoint: Checkpoint = Field(default_factory=Checkpoint)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
