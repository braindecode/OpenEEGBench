#!/usr/bin/env python
"""Download all datasets from HuggingFace Hub, one at a time to avoid OOM."""

import gc
import os
os.environ.setdefault("HF_HOME", "/expanse/projects/nemar/eeg_finetuning/pierre/hf_cache")

TOKEN = "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"

DATASETS = [
    "braindecode/chbmit",
    "braindecode/isruc-sleep",
    "braindecode/mdd_mumtaz2016",
    "braindecode/seed-v",
    "braindecode/seed-vig",
    "braindecode/tuab",
    "braindecode/tuev",
    # Already downloaded:
    # "braindecode/arithmetic_zyma2019",
    # "braindecode/bcic2a",
    # "braindecode/bcic2020-3",
    # "braindecode/physionet",
    # "braindecode/faced",
]

for ds_id in DATASETS:
    print(f"Downloading {ds_id}...", flush=True)
    try:
        from braindecode.datasets import BaseConcatDataset
        ds = BaseConcatDataset.pull_from_hub(ds_id, token=TOKEN)
        print(f"  OK: {len(ds)} windows", flush=True)
        del ds
        gc.collect()
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

print("All done!")
