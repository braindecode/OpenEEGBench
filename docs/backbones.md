# Built-in Backbones

Open EEG Bench ships with the following pretrained EEG foundation models:

| Backbone | Reference | Pretrained Source | Normalization |
|----------|-----------|-------------------|---------------|
| **BIOT** | [Yang et al. 2023](https://arxiv.org/abs/2305.10351) | [`braindecode/biot-pretrained-prest-16chs`](https://huggingface.co/braindecode/biot-pretrained-prest-16chs) | Percentile scale (q=95) |
| **LaBraM** | [Jiang et al. 2024](https://arxiv.org/abs/2405.18765) | [`braindecode/labram-pretrained`](https://huggingface.co/braindecode/labram-pretrained) | Divide by 100 |
| **BENDR** | [Kostas et al. 2021](https://doi.org/10.3389/fnhum.2021.653659) | [`braindecode/braindecode-bendr`](https://huggingface.co/braindecode/braindecode-bendr) | Min-max scale [-1, 1] |
| **CBraMod** | [Wang et al. 2025](https://arxiv.org/abs/2412.07236) | [`braindecode/cbramod-pretrained`](https://huggingface.co/braindecode/cbramod-pretrained) | Divide by 100 |
| **SignalJEPA** | [Guetschel et al. 2024](https://arxiv.org/abs/2403.11772) | [`braindecode/SignalJEPA-pretrained`](https://huggingface.co/braindecode/SignalJEPA-pretrained) | None |
| **REVE** | [Music et al. 2025](https://arxiv.org/abs/2510.21585) | [`brain-bzh/reve-base`](https://huggingface.co/brain-bzh/reve-base) | Z-score (clip σ=15) |

These are available as factory functions in `open_eeg_bench.default_configs.backbones`:

```python
from open_eeg_bench.default_configs.backbones import biot, labram, bendr, cbramod, signaljepa, reve
```
