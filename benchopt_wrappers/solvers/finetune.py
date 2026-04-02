from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from open_eeg_bench.experiment import Experiment
    from open_eeg_bench.default_configs import (
        ALL_BACKBONES,
        ALL_FINETUNING,
        ALL_HEADS,
    )
    from open_eeg_bench.default_configs.experiments import default_training


class Solver(BaseSolver):
    name = "FineTune"

    sampling_strategy = "run_once"

    parameters = {
        "backbone_name": [
            "biot",
            "labram",
            "bendr",
            "cbramod",
            "signal_jepa",
            "reve",
        ],
        "finetuning_name": [
            "frozen",
            # "lora",
            # "ia3",
            # "adalora",
            # "dora",
            # "oft",
            "full_finetune",
            # "two_stages",
        ],
        "head_name": [
            "linear_head",
            # "mlp_head",
            # "original_head",
        ],
        "max_epochs": [30],  # For testing purposes, set to 1 epoch. Adjust as needed.
    }

    test_config = {
        "backbone_name": "biot",
        "finetuning_name": "frozen",
        "head_name": "linear_head",
        "max_epochs": 1,
    }

    def set_objective(self, dataset_config):
        self.dataset_config = dataset_config
        self.seed = self.get_seed(use_repetition=True)

    def run(self, _):
        backbone = ALL_BACKBONES[self.backbone_name]()
        finetuning = ALL_FINETUNING[self.finetuning_name]()
        head = ALL_HEADS[self.head_name]()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        training = default_training().model_copy(update={"device": device})
        training.max_epochs = self.max_epochs

        exp = Experiment(
            seed=self.seed,
            backbone=backbone,
            head=head,
            finetuning=finetuning,
            dataset=self.dataset_config,
            training=training,
        )
        self.result = exp.run()

    def get_result(self):
        return self.result
