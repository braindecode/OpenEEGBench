"""Classification / regression head configurations.

A Head replaces the backbone's final layer with a new one suited to the
downstream task.  ``OriginalHead`` keeps the model's built-in head.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import torch.nn as nn

log = logging.getLogger(__name__)


class LinearHead(BaseModel):
    """Replace the head with a single linear layer."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["linear"] = "linear"

    def apply(self, model, n_outputs: int, head_module_name: str) -> None:
        import torch.nn as nn

        new_head = nn.Sequential(nn.Flatten(1), nn.LazyLinear(n_outputs))
        setattr(model, head_module_name, new_head)
        log.info(
            "Replaced '%s' with LinearHead (n_outputs=%d)", head_module_name, n_outputs
        )


class MLPHead(BaseModel):
    """Replace the head with a two-layer MLP."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["mlp"] = "mlp"
    hidden_dim: int = 256
    dropout: float = 0.5

    def apply(self, model, n_outputs: int, head_module_name: str) -> None:
        import torch.nn as nn

        new_head = nn.Sequential(
            nn.Flatten(1),
            nn.LazyLinear(self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, n_outputs),
        )
        setattr(model, head_module_name, new_head)
        log.info(
            "Replaced '%s' with MLPHead (hidden=%d, dropout=%.2f)",
            head_module_name,
            self.hidden_dim,
            self.dropout,
        )


class OriginalHead(BaseModel):
    """Keep the backbone's original head unchanged."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["original"] = "original"

    def apply(self, model: nn.Module, n_outputs: int, head_module_name: str) -> None:
        log.info("Keeping original head '%s'", head_module_name)


class FlattenHead(BaseModel):
    """Replace the head with nn.Flatten(1). For ridge probing only."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["flatten"] = "flatten"

    def apply(self, model, n_outputs: int, head_module_name: str) -> None:
        import torch.nn as nn

        setattr(model, head_module_name, nn.Flatten(1))
        log.info("Replaced '%s' with FlattenHead", head_module_name)


Head = Annotated[
    Union[LinearHead, MLPHead, OriginalHead, FlattenHead],
    Field(discriminator="kind"),
]
