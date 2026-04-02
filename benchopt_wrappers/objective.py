from benchopt import BaseObjective, safe_import_context
from pathlib import Path


with safe_import_context() as import_ctx:
    import open_eeg_bench  # noqa: F401

# Install local files instead of relying on pypi version
OpenEEGBench = Path(__file__).parents[1]


class Objective(BaseObjective):
    name = "EEG-Bench"
    url = "https://github.com/OpenEEG-Bench/open-eeg-bench"

    min_benchopt_version = "1.9"

    # Install local repository as a pip package
    requirements = [f"pip::{OpenEEGBench}"]
    sampling_strategy = "run_once"

    def set_data(self, dataset_config):
        self.dataset_config = dataset_config

    def get_objective(self):
        return dict(dataset_config=self.dataset_config)

    def evaluate_result(self, **result):
        out = {}
        if "test_balanced_accuracy" in result:
            out["balanced_accuracy"] = result["test_balanced_accuracy"]
        elif "test_r2" in result:
            out["r2"] = result["test_r2"]

        adapter_stats = result.get("adapter_stats", {})
        for k, v in adapter_stats.items():
            if isinstance(v, (int, float)):
                out[f"adapter_{k}"] = float(v)

        return out

    def get_one_result(self):
        return dict(
            test_balanced_accuracy=0.0,
            adapter_stats={
                "method": "test",
                "total_params": 0,
                "trainable_params": 0,
                "trainable_pct": 0.0,
            },
        )
