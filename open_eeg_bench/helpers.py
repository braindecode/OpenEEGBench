import time
from typing import TYPE_CHECKING, ClassVar
import os
import contextlib
import logging
import math
from operator import attrgetter
from joblib import Parallel, delayed


if TYPE_CHECKING:
    import pandas as pd

from exca.helpers import to_config_model
from pydantic import BaseModel, model_validator, ConfigDict
from exca import TaskInfra


from open_eeg_bench.experiment import Experiment, collect_completed_results, run_many

log = logging.getLogger(__name__)


class MetaExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiments: list[Experiment]
    n_jobs: int = -1

    _exclude_from_cls_uid: ClassVar[tuple[str, ...]] = ("n_jobs",)
    infra: TaskInfra = TaskInfra()

    @model_validator(mode="after")
    def validate_experiments(self):
        for exp in self.experiments:
            assert exp.infra.cluster is None
            assert exp.infra.folder is not None
        return self

    @staticmethod
    def worker_function(exp: Experiment):
        try:
            return exp.run()
        except Exception as e:
            print(f"Error with experiment {exp}: {e}")
            return None

    @infra.apply()
    def run(self) -> list[dict | None]:
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.worker_function)(exp) for exp in self.experiments
        )
        return list(results)


def run_many_with_queue(
    *,
    experiments: list[Experiment | MetaExperiment],
    queue_size: int = 20,
    sleep_seconds: float = 5.0,
) -> None:
    """Run many experiments with a queue to avoid overloading the cluster scheduler.
    This is useful when you have a large number of experiments to run, but don't want to submit them all at once to the cluster scheduler (e.g., SLURM) to avoid overloading it.

    .. warning::
        This function is experimental and not stable. Its interface and
        behavior may change without notice in future versions.

    To launch this function as a SLURM job, you can decorate it
    with exca's helper:

    ```python
    with_infra(cluster="slurm", ...)(run_many_with_queue)(experiments=all_experiments)
    ```

    Parameters
    ----------
    experiments: list[Experiment]
        The list of experiments to run.
    queue_size: int
        The maximum number of experiments to keep in the queue (running or pending) at any time.
    sleep_seconds: float
        The number of seconds to sleep between each check of the queue.
    """
    import submitit.helpers

    _original_clean_env = submitit.helpers.clean_env
    # SLURM_CONF = "/cm/shared/apps/slurm/var/etc/expanse/slurm.conf"

    @contextlib.contextmanager
    def _clean_env_preserve_conf(*args, **kwargs):
        slurm_conf = os.environ.get("SLURM_CONF")
        with _original_clean_env(*args, **kwargs):
            if slurm_conf is not None:
                os.environ["SLURM_CONF"] = slurm_conf
            yield

    submitit.helpers.clean_env = _clean_env_preserve_conf

    experiments_to_launch = list(experiments)
    experiments_in_progress = []

    n_total = len(experiments)
    n_finished = 0
    n_launched = 0
    while experiments_to_launch:

        # empty the queue:
        # status can be ['not submitted', 'running', 'completed', 'failed']
        still_in_progress = []
        for exp, job in experiments_in_progress:
            status = exp.infra.status()
            if status not in ["completed", "failed"]:
                still_in_progress.append((exp, job))
            else:
                n_finished += 1
                print(f"{n_finished}/{n_total} jobs finished: {job} (status {status})")
        experiments_in_progress = still_in_progress

        # fill-up the queue
        while len(experiments_in_progress) < queue_size and experiments_to_launch:
            exp = experiments_to_launch.pop(0)
            job = exp.infra.job()  # non-blocking launch
            experiments_in_progress.append((exp, job))
            n_launched += 1
            print(f"{n_launched}/{n_total} jobs launched: {job}")

        time.sleep(sleep_seconds)


def run_multiple_per_node(
    experiments: list[Experiment],
    max_experiments_per_node: int = 10,
    max_experiments_running_per_node: int = -1,
    only_return_configs: bool = False,
    max_workers: int = 256,
    sort_experiments: bool = False,
):
    """Group experiments and submit each group as a single SLURM job.

    .. warning::
        This function is experimental and not stable. Its interface and
        behavior may change without notice in future versions.

    Instead of one SLURM job per experiment, this packs up to
    ``max_experiments_per_node`` experiments into a single job.
    Each job runs its experiments locally (sequentially via ``run_many``).

    Parameters
    ----------
    experiments: list[Experiment]
        Experiments to run. Must all be configured with ``cluster="slurm"``.
    max_experiments_per_node: int
        Maximum number of experiments to run within a single SLURM job.
        If -1, all experiments are packed into a single job (no grouping).
    max_experiments_running_per_node: int
        Maximum number of experiments to run concurrently on the same node.
        If -1, it defaults to `max_experiments_per_node` (i.e., all experiments in the same job run concurrently).
    only_return_configs: bool
        If True, return the list of MetaExperiment without submitting them.
    max_workers: int
        Maximum number of SLURM jobs running concurrently.
    sort_experiments: bool
        Whether to sort experiments by dataset/backbone/finetuning/head before grouping them.
    """
    if len(experiments) == 0:
        return None

    # Figure out which experiments to skip
    skip_list = [False] * len(experiments)
    for i, exp in enumerate(experiments):
        status = exp.infra.status()
        if (
            (status == "not submitted")
            or (exp.infra.mode == "force")
            or ((status == "failed") and (exp.infra.mode == "retry"))
        ):
            continue
        skip_list[i] = True
        print(
            f"Skipping experiment {i} with status '{status}' and mode '{exp.infra.mode}'"
        )
    print(
        f"===\nSkipping {sum(skip_list)}/{len(experiments)} experiments in total.\n==="
    )

    # Save the SLURM infra, then switch experiments to local execution
    # (they will run locally *inside* the SLURM job), and filter out experiments
    # that are already complete or running.
    first = experiments[0]
    assert (
        first.infra.cluster == "slurm"
    ), "This function is designed for SLURM execution"
    original_infra = first.infra.model_dump(
        mode="python",
        exclude_computed_fields=True,
        exclude_defaults=True,
        exclude_unset=True,
    )
    experiments = [
        exp.infra.clone_obj({"infra": {"cluster": None}})
        for exp, skip in zip(experiments, skip_list)
        if not skip
    ]

    # Sort by dataset > backbone > finetuning > head so that experiments
    # with similar durations end up in the same group (freeing nodes sooner).
    if sort_experiments:
        experiments = sorted(
            experiments,
            key=attrgetter(
                "dataset.hf_id", "backbone.model_cls", "finetuning.kind", "head.kind"
            ),
        )

    # Create groups of experiments to run together on the same node
    n = max_experiments_per_node
    if n > 0:
        n_groups = math.ceil(len(experiments) / n)
        groups = [experiments[i * n : (i + 1) * n] for i in range(n_groups)]
    else:
        groups = [experiments]

    # Wrap each group into a MetaExperiment submitted as one SLURM job.
    meta_infra = dict(original_infra)
    meta_infra["mode"] = "force"
    meta_experiments = [
        MetaExperiment(
            experiments=group,
            infra=meta_infra,  # type: ignore
            n_jobs=max_experiments_running_per_node,
        )
        for group in groups
    ]
    print(
        f"Generated {len(meta_experiments)} meta-experiments which contain "
        f"{[len(me.experiments) for me in meta_experiments]} experiments each."
    )

    if only_return_configs:
        return meta_experiments

    with meta_experiments[0].infra.job_array(max_workers=max_workers) as array:
        array.extend(meta_experiments)
