"""Model registry, with the parameter-matching the protocol requires.

PROTOCOL.md §3 fixes the four models to within 10% on parameter count in Phase 1.
The widths below are the result of that matching, and `check_matched()` enforces
it — an unmatched grid would confound "more symmetry" with "more capacity", which
is precisely the confusion the project exists to remove.
"""
from __future__ import annotations

import torch.nn as nn

from .deepsets import DeepSets
from .particlenet import ParticleNetLite
from .lorentznet import LorentzNet

# M0 and M1 share an architecture; they differ only in training-time augmentation.
SPECS = {
    "M0": dict(cls=DeepSets,        kwargs=dict(hidden=156, latent=156), augment=False),
    "M1": dict(cls=DeepSets,        kwargs=dict(hidden=156, latent=156), augment=True),
    "M2": dict(cls=ParticleNetLite, kwargs=dict(hidden=100, k=8),        augment=False),
    "M3": dict(cls=LorentzNet,      kwargs=dict(h_dim=66, m_dim=66, n_blocks=3), augment=False),
}

# Which view of a jet each model consumes.
INPUT = {"M0": "features", "M1": "features", "M2": "features", "M3": "p4"}

# Phase 3 (PROTOCOL amendment A1): matched forward-pass FLOPs instead of matched
# parameters. Measured at P=100, M3 costs 1.069B FLOPs per example against M1's
# 10.1M — a factor of 106 at equal parameter count. Reaching M3's budget takes
# M1 at hidden=1396 (7.8M parameters against M3's 98k).
#
# The baseline is enlarged rather than the equivariant model shrunk: shrinking
# M3 would cripple the model under test and manufacture H3's predicted result.
FLOPS_MATCHED = {
    "M1": dict(hidden=1396, latent=1396),
}


def build(model_id: str, budget: str = "params") -> nn.Module:
    spec = SPECS[model_id]
    kwargs = spec["kwargs"]
    if budget == "flops" and model_id in FLOPS_MATCHED:
        kwargs = {**kwargs, **FLOPS_MATCHED[model_id]}
    return spec["cls"](**kwargs)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_matched(tolerance: float = 0.10) -> dict[str, int]:
    """Raise if the four models are not matched within `tolerance`."""
    counts = {k: n_params(build(k)) for k in SPECS}
    lo, hi = min(counts.values()), max(counts.values())
    if (hi - lo) / lo > tolerance:
        raise AssertionError(
            f"parameter counts not matched within {tolerance:.0%}: {counts}"
        )
    return counts
