from pathlib import Path
import time
from typing import TYPE_CHECKING
import os
import contextlib

from exca.helpers import with_infra
import submitit.helpers

if TYPE_CHECKING:
    import pandas as pd

from open_eeg_bench.experiment import Experiment, collect_completed_results


with_infra(
    folder=Path("~/.cache/exca/").expanduser(),
    cluster="slurm",
    mode="force",
    slurm_partition="shared",
    slurm_account="csd403",
    job_name="queue-launcher",
    nodes=1,
    cpus_per_task=1,
    mem_gb=4,
    timeout_min=60 * 24,
    slurm_additional_parameters={
        "qos": "shared-normal",
    },
)
def run_many_with_queue(
    *,
    experiments: list[Experiment],
    queue_size: int = 20,
    sleep_seconds: float = 5.0,
) -> "pd.DataFrame":
    """Run many experiments with a queue to avoid overloading the cluster scheduler.
    This is useful when you have a large number of experiments to run, but don't want to submit them all at once to the cluster scheduler (e.g., SLURM) to avoid overloading it.

    Parameters
    ----------
    experiments: list[Experiment]
        The list of experiments to run.
    queue_size: int
        The maximum number of experiments to keep in the queue (running or pending) at any time.
    sleep_seconds: float
        The number of seconds to sleep between each check of the queue.
    """

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
    experiments_in_progress: list[Experiment] = []

    while experiments_to_launch:

        # empty the queue:
        # status can be ['not submitted', 'running', 'completed', 'failed']
        experiments_in_progress = [
            exp
            for exp in experiments_in_progress
            if exp.infra.status() not in ["completed", "failed"]
        ]

        # fill-up the queue
        while len(experiments_in_progress) < queue_size and experiments_to_launch:
            exp = experiments_to_launch.pop(0)
            experiments_in_progress.append(exp)
            _ = exp.infra.job()  # non-blocking launch

        time.sleep(sleep_seconds)

    return collect_completed_results(experiments)
