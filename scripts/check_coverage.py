#!/usr/bin/env python
"""Check coverage of successful experiments."""
import os
import glob

results_dir = "/expanse/projects/nemar/eeg_finetuning/pierre/oeb_results/open_eeg_bench.experiment.ExperimentHandler.run,0/default"

success_files = []
for f in glob.glob(os.path.join(results_dir, "*.csv")):
    with open(f) as fh:
        if "balanced_accuracy" in fh.read():
            success_files.append(os.path.basename(f))

print(f"Total successful: {len(success_files)}")

checks = {
    "DATASETS": {
        "arithmetic_zyma2019": "arithmetic_zy",
        "bcic2a": "bcic2a",
        "bcic2020_3": "bcic2020",
        "physionet": "physionet",
        "chbmit": "chbmit",
        "faced": "faced",
        "isruc_sleep": "isruc",
        "mdd_mumtaz2016": "mdd_mumtaz",
        "seed_v": "seed-v",
        "seed_vig": "seed-vig",
        "tuab": "tuab",
        "tuev": "tuev",
    },
    "FINETUNING": {
        "lora": ["alpha=32", "kind=lora", "lora_drop"],
        "ia3": ["kind=ia3"],
        "adalora": ["adalora", "init_r=", "deltaT"],
        "dora": ["kind=dora", "use_dora"],
        "oft": ["block_s", "kind=oft", "coft"],
        "full": ["kind=full"],
        "frozen": ["kind=frozen"],
    },
    "HEADS": {
        "linear": "kind=linear",
        "mlp": "kind=mlp",
        "original": "kind=original",
    },
}

for section, mapping in checks.items():
    print(f"\n{section}:")
    for name, pats in mapping.items():
        if isinstance(pats, str):
            pats = [pats]
        n = sum(1 for f in success_files if any(p in f for p in pats))
        status = str(n) if n else "** MISSING **"
        print(f"  {name}: {status}")
