"""Benchmark every braindecode classifier trained from scratch on every dataset.

For each braindecode model, ``oeb.benchmark(..., from_scratch=True)`` is called
with ``only_return_configs=True`` to collect the (model × dataset × seed) configs
without launching anything. All configs are then submitted together via
``run_many`` as a single SLURM array, so each (model, dataset, seed) becomes one
SLURM job and ``max_workers`` controls how many run concurrently.

Adapt ``slurm_partition`` and ``slurm_account`` to your cluster.
"""

import open_eeg_bench as oeb

# Braindecode classifiers that build with the generic kwargs we forward
# (n_chans, n_outputs, n_times, sfreq, chs_info) and expose ``final_layer`` as
# the head module. Pretrained foundation models (BIOT, Labram, BENDR, CBraMod,
# REVE, SignalJEPA, EEGPT, LUNA) are intentionally omitted — they have factories
# in default_configs.backbones and are meant to be loaded with weights.
BRAINDECODE_SCRATCH_MODELS = [
    "braindecode.models.ATCNet",
    "braindecode.models.AttentionBaseNet",
    "braindecode.models.BDTCN",
    "braindecode.models.CTNet",
    "braindecode.models.ContraWR",
    "braindecode.models.DGCNN",
    "braindecode.models.Deep4Net",
    "braindecode.models.DeepSleepNet",
    "braindecode.models.EEGConformer",
    "braindecode.models.EEGITNet",
    "braindecode.models.EEGInceptionERP",
    "braindecode.models.EEGInceptionMI",
    "braindecode.models.EEGMiner",
    "braindecode.models.EEGNeX",
    "braindecode.models.EEGNet",
    "braindecode.models.EEGNetv4",
    "braindecode.models.EEGSimpleConv",
    "braindecode.models.EEGSym",
    "braindecode.models.EEGTCNet",
    "braindecode.models.FBCNet",
    "braindecode.models.FBLightConvNet",
    "braindecode.models.FBMSNet",
    "braindecode.models.HybridNet",
    "braindecode.models.IFNet",
    "braindecode.models.MEDFormer",
    "braindecode.models.MSVTNet",
    "braindecode.models.PBT",
    "braindecode.models.SCCNet",
    "braindecode.models.SPARCNet",
    "braindecode.models.ShallowFBCSPNet",
    "braindecode.models.SincShallowNet",
    "braindecode.models.SleepStagerChambon2018",
    "braindecode.models.SSTDPN",
    "braindecode.models.SyncNet",
    "braindecode.models.TIDNet",
    "braindecode.models.TSception",
]

INFRA = {
    "folder": "./results",
    "cluster": "slurm",
    "gpus_per_node": 1,
    "mem_gb": 32,
    "timeout_min": 240,
    "cpus_per_task": 4,
    "slurm_partition": "gpu",        # adapt to your cluster
    "slurm_account": "my_account",   # adapt to your cluster
}


def main() -> None:
    all_configs = []
    for model_cls in BRAINDECODE_SCRATCH_MODELS:
        configs = oeb.benchmark(
            model_cls=model_cls,
            from_scratch=True,
            datasets=None,        # all datasets
            n_seeds=3,
            device="cuda",
            infra=INFRA,
            only_return_configs=True,
        )
        all_configs.extend(configs)

    print(
        f"Submitting {len(all_configs)} jobs "
        f"({len(BRAINDECODE_SCRATCH_MODELS)} models × all datasets × seeds)"
    )

    results = oeb.experiment.run_many(
        all_configs,
        max_workers=64,   # concurrent SLURM jobs
        collect_all=True,
    )
    print(results)


if __name__ == "__main__":
    main()
