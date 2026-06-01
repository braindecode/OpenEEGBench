import numpy as np
from braindecode.models import EEGPT, InterpolatedModel
from braindecode.models.eegpt import EEGPT_CHANNELS
from mne.channels import make_standard_montage

montage = make_standard_montage("standard_1020")
ch_pos = {
    ch.upper(): (ch, loc) for ch, loc in montage.get_positions()["ch_pos"].items()
}
_EEGPT_TARGET_CHS_INFO = [
    {
        "ch_name": ch_pos[ch.upper()][0],
        "kind": "eeg",
        "loc": ch_pos[ch.upper()][1],
    }
    for ch in EEGPT_CHANNELS
]
InterpolatedEEGPT = InterpolatedModel(
    EEGPT, _EEGPT_TARGET_CHS_INFO, name="InterpolatedEEGPT"
)
