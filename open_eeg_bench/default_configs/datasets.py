"""Default dataset configurations for all supported downstream datasets.

Each function returns a Dataset config with the HuggingFace Hub ID,
number of classes, and a predefined train/val/test split.

All datasets are pre-windowed and hosted at https://huggingface.co/braindecode.

Note: Subject/session IDs may be strings or ints depending on the dataset.
The values here must match the metadata column in the HF dataset exactly.
"""

from open_eeg_bench.dataset import Dataset, PredefinedSplitter


def arithmetic_zyma2019(**overrides) -> Dataset:
    """Mental arithmetic vs. rest — 2 classes, 36 subjects."""
    defaults = dict(
        hf_id="braindecode/arithmetic_zyma2019",
        n_classes=2,
        splitter=PredefinedSplitter(
            metadata_key="subject",
            train_values=[f"{i:02d}" for i in range(28)],
            val_values=[f"{i:02d}" for i in range(28, 32)],
            test_values=[f"{i:02d}" for i in range(32, 36)],
        ),
        batch_size=64,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def bcic2a(**overrides) -> Dataset:
    """BCI Competition IV 2a — 4-class motor imagery, 9 subjects."""
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


def bcic2020_3(**overrides) -> Dataset:
    """BCI Competition 2020-3 — 5-class imagined speech."""
    defaults = dict(
        hf_id="braindecode/bcic2020-3",
        n_classes=5,
        splitter=PredefinedSplitter(
            metadata_key="run",
            train_values=["00"],
            val_values=["01"],
            test_values=["02"],
        ),
        batch_size=32,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def physionet(**overrides) -> Dataset:
    """PhysioNet Motor Imagery — 4 classes, 109 subjects."""
    defaults = dict(
        hf_id="braindecode/physionet",
        n_classes=4,
        splitter=PredefinedSplitter(
            metadata_key="subject",
            train_values=list(range(1, 71)),
            val_values=list(range(71, 90)),
            test_values=list(range(90, 110)),
        ),
        batch_size=32,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def chbmit(**overrides) -> Dataset:
    """CHB-MIT seizure detection — 2 classes, 23 subjects."""
    defaults = dict(
        hf_id="braindecode/chbmit",
        n_classes=2,
        splitter=PredefinedSplitter(
            metadata_key="subject",
            train_values=[f"chb{i:02d}" for i in range(1, 20)],
            val_values=["chb20", "chb21"],
            test_values=["chb22", "chb23"],
        ),
        batch_size=32,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def faced(**overrides) -> Dataset:
    """FACED emotion recognition — 9 classes, 123 subjects."""
    defaults = dict(
        hf_id="braindecode/faced",
        n_classes=9,
        splitter=PredefinedSplitter(
            metadata_key="subject",
            train_values=[f"{i:03d}" for i in range(0, 80)],
            val_values=[f"{i:03d}" for i in range(80, 100)],
            test_values=[f"{i:03d}" for i in range(100, 123)],
        ),
        batch_size=32,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def isruc_sleep(**overrides) -> Dataset:
    """ISRUC sleep staging — 5 classes, ~60 usable subjects."""
    defaults = dict(
        hf_id="braindecode/isruc-sleep",
        n_classes=5,
        splitter=PredefinedSplitter(
            metadata_key="subject",
            # fmt: off
            train_values=[
                "I011", "I013", "I014", "I017", "I019", "I020",
                "I021", "I023", "I025", "I027", "I028", "I029", "I030",
                "I031", "I032", "I033", "I034", "I035", "I036", "I037", "I038", "I039",
                "I041", "I042", "I043", "I044", "I045", "I046", "I047", "I048", "I049", "I050",
                "I051", "I052", "I053", "I054", "I055", "I056", "I057", "I058", "I059", "I060",
                "I061", "I062", "I063", "I064", "I065", "I066", "I067", "I068", "I069", "I070",
                "I071", "I072", "I073", "I074", "I075", "I076", "I077", "I078", "I079", "I080",
            ],
            # fmt: on
            val_values=[f"I{i:03d}" for i in range(81, 91)],
            test_values=[f"I{i:03d}" for i in range(91, 101)],
        ),
        batch_size=32,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def mdd_mumtaz2016(**overrides) -> Dataset:
    """MDD detection (Mumtaz 2016) — 2 classes."""
    defaults = dict(
        hf_id="braindecode/mdd_mumtaz2016",
        n_classes=2,
        splitter=PredefinedSplitter(
            metadata_key="subject",
            # fmt: off
            train_values=[
                "HS1", "HS10", "HS11", "HS12", "HS13", "HS14", "HS15", "HS16",
                "HS17", "HS18", "HS19", "HS2", "HS20", "HS21", "HS22",
                "MDDS1", "MDDS10", "MDDS11", "MDDS12", "MDDS13", "MDDS14",
                "MDDS15", "MDDS16", "MDDS17", "MDDS18", "MDDS19", "MDDS2",
                "MDDS20", "MDDS21",
            ],
            val_values=["HS23", "HS24", "HS25", "MDDS22", "MDDS23", "MDDS24", "MDDS25"],
            test_values=[
                "HS26", "HS27", "HS28", "HS29", "HS3", "HS30", "HS4", "HS5",
                "HS6", "HS7", "HS8", "HS9",
                "MDDS26", "MDDS27", "MDDS28", "MDDS29", "MDDS3", "MDDS30",
                "MDDS31", "MDDS32", "MDDS33", "MDDS34", "MDDS4", "MDDS5",
                "MDDS6", "MDDS7", "MDDS8", "MDDS9",
            ],
            # fmt: on
        ),
        batch_size=32,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def seed_v(**overrides) -> Dataset:
    """SEED-V emotion recognition — 5 classes, session-based split."""
    defaults = dict(
        hf_id="braindecode/seed-v",
        n_classes=5,
        splitter=PredefinedSplitter(
            metadata_key="session",
            train_values=["01"],
            val_values=["02"],
            test_values=["03"],
        ),
        batch_size=32,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def seed_vig(**overrides) -> Dataset:
    """SEED-VIG vigilance regression — continuous perclos target."""
    defaults = dict(
        hf_id="braindecode/seed-vig",
        n_classes=None,
        regression=True,
        splitter=PredefinedSplitter(
            metadata_key="subject",
            train_values=[f"{i:02d}" for i in [1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]],
            val_values=[f"{i:02d}" for i in [4, 5]],
            test_values=[f"{i:02d}" for i in [6, 7, 8, 9]],
        ),
        batch_size=32,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def tuab(**overrides) -> Dataset:
    """TUH Abnormal EEG — 2 classes (normal/abnormal), predefined train/eval split."""
    defaults = dict(
        hf_id="braindecode/tuab",
        n_classes=2,
        splitter=PredefinedSplitter(
            metadata_key="train",
            train_values=[True],
            val_values=[False],  # use eval set as val; override for CV
        ),
        batch_size=32,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


def tuev(**overrides) -> Dataset:
    """TUH EEG Events — 6 classes, predefined train/eval split."""
    defaults = dict(
        hf_id="braindecode/tuev",
        n_classes=6,
        splitter=PredefinedSplitter(
            metadata_key="split",
            train_values=["train"],
            val_values=["eval"],  # use eval set as val; override for CV
        ),
        batch_size=32,
    )
    defaults.update(overrides)
    return Dataset(**defaults)


ALL_DATASETS = {
    "arithmetic_zyma2019": arithmetic_zyma2019,
    "bcic2a": bcic2a,
    "bcic2020_3": bcic2020_3,
    "physionet": physionet,
    "chbmit": chbmit,
    "faced": faced,
    "isruc_sleep": isruc_sleep,
    "mdd_mumtaz2016": mdd_mumtaz2016,
    "seed_v": seed_v,
    "seed_vig": seed_vig,
    "tuab": tuab,
    "tuev": tuev,
}
