from __future__ import annotations

"""High-level benchmarking entry point.

``benchmark()`` is the simplest way to evaluate an EEG foundation model.
It accepts plain Python types (strings, dicts, lists) and returns a
:class:`~pandas.DataFrame` with one row per (dataset, finetuning) combination.
"""

from typing import TYPE_CHECKING
import logging
from typing import Any

if TYPE_CHECKING:
    import pandas as pd
    from torch import nn

from open_eeg_bench.experiment import Experiment, run_many
from open_eeg_bench.default_configs.experiments import make_all_experiments
from open_eeg_bench.backbone import PretrainedBackbone

log = logging.getLogger(__name__)


def benchmark(
    model_cls: type["nn.Module"] | str,
    *,
    hub_repo: str | None = None,
    checkpoint_url: str | None = None,
    checkpoint_path: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    peft_target_modules: list[str] | None = None,
    head_module_name: str = "final_layer",
    peft_ff_modules: list[str] | None = None,
    normalization: Any | None = None,
    datasets: list[str] | None = None,
    heads: list[str] | None = None,
    finetuning_strategies: list[str] | None = None,
    n_seeds: int = 3,
    infra: dict[str, Any] | None = None,
) -> "pd.DataFrame":
    """Benchmark an EEG model on multiple datasets and finetuning strategies.

    This is the main entry point for evaluating a new model. All config
    classes are created internally — you only need plain Python types.

    Parameters
    ----------
    model_cls : type["nn.Module"] | str
        The model class or its dotted import path (string),
        e.g. ``"braindecode.models.BIOT"``.
    hub_repo : str, optional
        HuggingFace Hub repo ID for pretrained weights.
    checkpoint_url : str, optional
        Direct URL to pretrained weights.
    checkpoint_path : str, optional
        Local path to pretrained weights.
    model_kwargs : dict, optional
        Keyword arguments forwarded to the model constructor.
    peft_target_modules : list[str], optional
        Module names to target for PEFT adapters (LoRA, etc.).
        Not needed when only using ``"frozen"`` finetuning.
    head_module_name : str
        Name of the classification head module (default: ``"final_layer"``).
    peft_ff_modules : list[str], optional
        Feedforward module names for IA3 adapter.
    normalization : Normalization, optional
        Post-window normalization to apply to each data window.
    datasets : list[str], optional
        Dataset names to evaluate on. If ``None``, uses all datasets.
        Valid names: ``"arithmetic_zyma2019"``, ``"bcic2a"``,
        ``"bcic2020_3"``, ``"physionet"``, ``"chbmit"``, ``"faced"``,
        ``"isruc_sleep"``, ``"mdd_mumtaz2016"``, ``"seed_v"``,
        ``"seed_vig"``, ``"tuab"``, ``"tuev"``.
    finetuning_strategies : list[str], optional
        Finetuning strategy names. If ``None``, uses ``["frozen"]``
        (linear probing). Valid names: ``"frozen"``, ``"lora"``,
        ``"ia3"``, ``"adalora"``, ``"dora"``, ``"oft"``,
        ``"full_finetune"``.
    heads : list[str], optional
        Head names to evaluate. If ``None``, uses ``["linear_head", "mlp_head"]``.
        Valid names: ``"linear_head"``, ``"mlp_head"``, ``"original_head"``.
    n_seeds : int
        Number of random seeds for initialization of the heads and new layers.
    infra : dict, optional
        Infrastructure config passed to each experiment's ``infra`` field.
        Controls caching and execution. Example::

            {"folder": "./results"}                     # local, cached
            {"folder": "./results", "cluster": "slurm"} # SLURM submission

    Returns
    -------
    pd.DataFrame
        Results with one row per (dataset, finetuning) combination.
        Columns include the test metric (``test_balanced_accuracy`` or
        ``test_r2``), adapter statistics, and ``error`` if the run failed.

    Examples
    --------
    Minimal — linear probing on all datasets::

        from open_eeg_bench import benchmark

        results = benchmark(
            model_cls="my_package.MyModel",
            checkpoint_url="https://my-weights.pth",
        )

    Pick specific datasets and add LoRA::

        results = benchmark(
            model_cls="my_package.MyModel",
            checkpoint_url="https://my-weights.pth",
            peft_target_modules=["encoder.linear1", "encoder.linear2"],
            datasets=["arithmetic_zyma2019", "bcic2a"],
            finetuning_strategies=["frozen", "lora"],
        )
    """
    backbone = PretrainedBackbone(
        model_cls=model_cls,
        hub_repo=hub_repo,
        checkpoint_url=checkpoint_url,
        checkpoint_path=checkpoint_path,
        model_kwargs=model_kwargs or {},
        peft_target_modules=peft_target_modules or [],
        head_module_name=head_module_name,
        peft_ff_modules=peft_ff_modules or [],
        normalization=normalization,
    )

    experiments = make_all_experiments(
        datasets=datasets,
        heads=heads,
        finetuning_strategies=finetuning_strategies,
        n_seeds=n_seeds,
    )

    update = {"backbone": backbone}
    if infra is not None:
        update["infra"] = infra
    experiments = [exp.model_copy(update=update) for exp in experiments]
    experiments = [Experiment.model_validate(exp) for exp in experiments]

    return run_many(experiments)
