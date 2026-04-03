import time
from typing import TYPE_CHECKING
import os
import contextlib
import logging
import math
from operator import attrgetter


if TYPE_CHECKING:
    import pandas as pd

from exca.helpers import to_config_model


from open_eeg_bench.experiment import Experiment, collect_completed_results, run_many

log = logging.getLogger(__name__)

MetaExperiment = to_config_model(run_many)


def run_many_with_queue(
    *,
    experiments: list[Experiment | MetaExperiment],
    queue_size: int = 20,
    sleep_seconds: float = 5.0,
) -> None:
    """Run many experiments with a queue to avoid overloading the cluster scheduler.
    This is useful when you have a large number of experiments to run, but don't want to submit them all at once to the cluster scheduler (e.g., SLURM) to avoid overloading it.

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
    only_return_configs: bool = False,
    max_workers: int = 256,
):
    if len(experiments) == 0:
        return None

    # 1. Modify the infras such that they are executed on the local node
    first = experiments[0]
    assert (
        first.infra.cluster == "slurm"
    ), "This function is designed for SLURM execution"
    original_infra = first.infra.model_dump(mode="python")
    experiments = [
        exp.infra.clone_obj({"infra": {"cluster": "local"}}) for exp in experiments
    ]

    # 2. Split experiments into groups of max_experiments_per_node
    # The preferred grouping is dataset > backbone > finetuning > head
    # We group them like this to try to group them by duration such that
    # within one group, all experiments roughly end (and free the node) simultaneously.
    sorted_experiments = sorted(
        experiments,
        key=attrgetter(
            "dataset.hf_id", "backbone.model_cls", "finetuning.kind", "head.kind"
        ),
    )
    n_groups = math.ceil(len(experiments) / max_experiments_per_node)
    n = max_experiments_per_node
    n_groups = math.ceil(len(sorted_experiments) / n)
    groups = [sorted_experiments[i * n : (i + 1) * n] for i in range(n_groups)]

    # 3. Create one meta-experiment per group
    meta_experiments = [
        MetaExperiment(
            experiments=group, wait=True, infra=original_infra  # type: ignore
        )
        for group in groups
    ]

    if only_return_configs:
        return meta_experiments

    with meta_experiments[0].infra.job_array(max_workers=max_workers) as array:
        array.extend(meta_experiments)
