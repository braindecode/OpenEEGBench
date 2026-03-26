"""Default backbone configurations for all supported architectures."""

from open_eeg_bench.backbone import (
    BENDRBackbone,
    BIOTBackbone,
    CBraModBackbone,
    LabramBackbone,
    REVEBackbone,
    SignalJEPABackbone,
)


def biot(**overrides) -> BIOTBackbone:
    return BIOTBackbone(**overrides)


def labram(**overrides) -> LabramBackbone:
    return LabramBackbone(**overrides)


def bendr(**overrides) -> BENDRBackbone:
    return BENDRBackbone(**overrides)


def cbramod(**overrides) -> CBraModBackbone:
    return CBraModBackbone(**overrides)


def signal_jepa(**overrides) -> SignalJEPABackbone:
    return SignalJEPABackbone(**overrides)


def reve(**overrides) -> REVEBackbone:
    return REVEBackbone(**overrides)


ALL_BACKBONES = {
    "biot": biot,
    "labram": labram,
    "bendr": bendr,
    "cbramod": cbramod,
    "signal_jepa": signal_jepa,
    "reve": reve,
}
