"""Default dataset configurations for all supported downstream datasets."""

from open_eeg_bench.dataset import Dataset, PredefinedSplitter, RandomSplitter


def arithmetic_zyma2019(**overrides) -> Dataset:
    defaults = dict(
        hf_id="braindecode/arithmetic_zyma2019",
        n_classes=2,
        splitter=PredefinedSplitter(
            metadata_key="subject",
            # 36 subjects with string IDs "00" to "35"
            train_values=[f"{i:02d}" for i in range(28)],
            val_values=[f"{i:02d}" for i in range(28, 32)],
            test_values=[f"{i:02d}" for i in range(32, 36)],
        ),
        batch_size=64,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def bcic2a(**overrides) -> Dataset:
    defaults = dict(
        hf_id="braindecode/bcic2a",
        n_classes=4,
        splitter=PredefinedSplitter(
            metadata_key="subject",
            train_values=[1, 2, 3],
            val_values=[4, 5, 6],
            test_values=[7, 8, 9],
        ),
        batch_size=64,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def physionet(**overrides) -> Dataset:
    defaults = dict(
        hf_id="braindecode/physionet",
        n_classes=4,
        splitter=RandomSplitter(val_split=0.2, test_split=0.1),
        batch_size=64,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


ALL_DATASETS = [arithmetic_zyma2019, bcic2a, physionet]
