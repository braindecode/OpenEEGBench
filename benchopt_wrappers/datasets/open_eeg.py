from benchopt import BaseDataset

from open_eeg_bench.default_configs.datasets import ALL_DATASETS


class Dataset(BaseDataset):
    name = "OpenEEG"

    parameters = {
        "dataset_name": [
            "arithmetic_zyma2019",
            "bcic2a",
            "bcic2020_3",
            "physionet",
            "chbmit",
            "faced",
            "isruc_sleep",
            "mdd_mumtaz2016",
            "seed_v",
            "seed_vig",
            "tuab",
            "tuev",
        ],
    }

    test_parameters = {
        "dataset_name": ["arithmetic_zyma2019"],
    }

    def get_data(self):
        dataset_config = ALL_DATASETS[self.dataset_name]()
        return dict(dataset_config=dataset_config)
