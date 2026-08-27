"""Top Quark Tagging Reference Dataset — loading and feature construction.

Kasieczka, Plehn, Thompson & Russell (2019), Zenodo 2603256. Public benchmark:
1.2M/400k/400k jets, each stored as 200 constituent four-momenta (E, px, py, pz)
ordered by decreasing pT, zero-padded.

Two views of the same jet are produced, because the models under comparison
disagree about what a jet *is*:

- `features`  — the standard engineered set (log pT, Δη, Δφ, ...). What Deep Sets
                and ParticleNet consume. Already Lorentz-*invariant* in part,
                which is exactly the confound this project measures.
- `p4`        — raw four-momenta. What LorentzNet consumes, and what any Lorentz
                transformation acts on.

Keeping both in one place means the boost applied for augmentation and the boost
applied to break the symmetry are literally the same code path.
"""
from __future__ import annotations

import numpy as np
import torch

MAX_CONSTITUENTS = 200
EPS = 1e-8


def _read_h5(path: str, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (p4, label) with p4 shaped (N, P, 4) in (E, px, py, pz) order."""
    import pandas as pd

    df = pd.read_hdf(path, key="table", stop=n)
    cols = [f"{v}_{i}" for i in range(MAX_CONSTITUENTS) for v in ("E", "PX", "PY", "PZ")]
    p4 = df[cols].to_numpy(dtype=np.float32).reshape(-1, MAX_CONSTITUENTS, 4)
    y = df["is_signal_new"].to_numpy(dtype=np.int64)
    return p4, y


def mask_of(p4: torch.Tensor) -> torch.Tensor:
    """True where a constituent is real rather than zero padding."""
    return p4.abs().sum(-1) > 0


def kinematics(p4: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(pt, eta, phi, e) from four-momenta, padding-safe."""
    e, px, py, pz = p4.unbind(-1)
    pt = torch.sqrt(px * px + py * py + EPS)
    eta = torch.asinh(pz / pt)
    phi = torch.atan2(py, px)
    return pt, eta, phi, e


def features(p4: torch.Tensor) -> torch.Tensor:
    """Engineered per-constituent features, relative to the jet axis.

    Seven channels, the conventional set for this benchmark:
      Δη, Δφ, log pT, log E, log(pT/pT_jet), log(E/E_jet), ΔR

    Note what this representation already does: by subtracting the jet axis it
    removes the longitudinal boost and the azimuthal rotation by hand. A model
    fed these features has been handed part of the symmetry for free — which is
    why "unconstrained baseline" is a slippery label, and why the honest
    comparison in this project is against augmentation rather than against a
    strawman.
    """
    mask = mask_of(p4)
    pt, eta, phi, e = kinematics(p4)

    jet = p4.sum(dim=1)                                   # (B, 4)
    jpt, jeta, jphi, je = kinematics(jet.unsqueeze(1))
    jpt, jeta, jphi, je = jpt.squeeze(1), jeta.squeeze(1), jphi.squeeze(1), je.squeeze(1)

    d_eta = eta - jeta.unsqueeze(1)
    d_phi = torch.remainder(phi - jphi.unsqueeze(1) + torch.pi, 2 * torch.pi) - torch.pi
    d_r = torch.sqrt(d_eta**2 + d_phi**2 + EPS)

    f = torch.stack([
        d_eta,
        d_phi,
        torch.log(pt + EPS),
        torch.log(e + EPS),
        torch.log(pt / (jpt.unsqueeze(1) + EPS) + EPS),
        torch.log(e / (je.unsqueeze(1) + EPS) + EPS),
        d_r,
    ], dim=-1)

    return f * mask.unsqueeze(-1)


class JetDataset(torch.utils.data.Dataset):
    """N jets, exposing both views. `n` is the pre-registered training-set size."""

    def __init__(self, path: str, n: int | None = None, seed: int = 0):
        p4, y = _read_h5(path, n=None)
        if n is not None and n < len(y):
            # Subsample with a fixed seed rather than taking the head: the file is
            # ordered, and a prefix is not a random sample of it.
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(y), size=n, replace=False)
            p4, y = p4[idx], y[idx]
        self.p4 = torch.from_numpy(p4)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.p4[i], self.y[i]
