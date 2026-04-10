"""Quick start: benchmark a built-in backbone on a single dataset.

This is the simplest way to verify that the benchmark runs end-to-end.
It evaluates BIOT with frozen linear probing on arithmetic_zyma2019.
"""

import open_eeg_bench as oeb

results = oeb.benchmark(
    model_cls="braindecode.models.BIOT",
    hub_repo="braindecode/biot-pretrained-prest-16chs",
    model_kwargs=dict(
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        drop_prob=0.5,
        att_drop_prob=0.2,
        att_layer_drop_prob=0.2,
        hop_length=100,
        max_seq_len=1024,
        return_feature=False,
    ),
    peft_target_modules=["to_q", "to_k", "to_v", "to_out", "w1", "w2"],
    datasets=["arithmetic_zyma2019"],
    n_seeds=1,
    device="cpu",
)

print(results)
