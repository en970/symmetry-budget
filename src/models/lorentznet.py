"""LorentzNet-style equivariant network. M3, the model under test.

Follows Gong, Gao, Shen et al. (2022), "An Efficient Lorentz Equivariant Graph
Neural Network for Jet Tagging". The construction is the whole point of the
project, so the guarantee is stated explicitly:

    Node four-vectors are only ever updated by adding linear combinations of
    *differences of four-vectors*, weighted by coefficients computed from
    Lorentz *invariants* (Minkowski norms and inner products). Since
        Λ(x_i + Σ_j c_ij (x_i - x_j)) = Λx_i + Σ_j c_ij (Λx_i - Λx_j)
    and the c_ij are unchanged by Λ, every block commutes with the Lorentz
    group, and a scalar readout of the final invariants is Lorentz invariant.

That property is verified numerically in `tests/test_equivariance.py`, not
assumed. A silently broken equivariance would make M3 an ordinary GNN with a
strange parameterisation, and the entire comparison would be measuring nothing.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.symmetry import minkowski
from .deepsets import mlp


def psi(x: torch.Tensor) -> torch.Tensor:
    """sgn(x)·log(|x|+1) — compresses the enormous dynamic range of s-channel
    invariants without discarding sign, which carries the spacelike/timelike
    distinction."""
    return torch.sign(x) * torch.log(x.abs() + 1.0)


class LGEB(nn.Module):
    """One Lorentz Group Equivariant Block."""

    def __init__(self, h_dim: int, m_dim: int, c: float = 1.0):
        super().__init__()
        self.c = c
        self.phi_e = mlp([2 * h_dim + 2, m_dim, m_dim])
        self.phi_x = mlp([m_dim, m_dim, 1], final_act=False)
        self.phi_h = mlp([h_dim + m_dim, h_dim, h_dim], final_act=False)
        self.phi_m = nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid())

    def forward(self, h: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        B, P, _ = h.shape
        pair = mask.unsqueeze(1) & mask.unsqueeze(2)                     # (B,P,P)
        pair = pair & ~torch.eye(P, dtype=torch.bool, device=h.device)

        diff = x.unsqueeze(2) - x.unsqueeze(1)                           # (B,P,P,4)
        norm = psi(minkowski(diff, diff))                                # (B,P,P) invariant
        inner = psi(minkowski(x.unsqueeze(2).expand(B, P, P, 4),
                              x.unsqueeze(1).expand(B, P, P, 4)))        # invariant

        hi = h.unsqueeze(2).expand(B, P, P, h.shape[-1])
        hj = h.unsqueeze(1).expand(B, P, P, h.shape[-1])
        m = self.phi_e(torch.cat([hi, hj, norm.unsqueeze(-1), inner.unsqueeze(-1)], dim=-1))
        m = m * pair.unsqueeze(-1)

        # Equivariant update: scalar coefficients times four-vector differences.
        coeff = self.phi_x(m) * pair.unsqueeze(-1)                       # (B,P,P,1)
        x = x + self.c * (coeff * diff).sum(dim=2) / pair.sum(dim=2, keepdim=True).clamp(min=1)

        # Invariant update.
        w = self.phi_m(m) * pair.unsqueeze(-1)
        agg = (w * m).sum(dim=2)
        h = h + self.phi_h(torch.cat([h, agg], dim=-1))
        return h * mask.unsqueeze(-1), x * mask.unsqueeze(-1)


class LorentzNet(nn.Module):
    def __init__(self, n_scalars: int = 1, h_dim: int = 72, m_dim: int = 72,
                 n_blocks: int = 3, n_classes: int = 2):
        super().__init__()
        self.embed = nn.Linear(n_scalars, h_dim)
        self.blocks = nn.ModuleList([LGEB(h_dim, m_dim) for _ in range(n_blocks)])
        self.head = mlp([h_dim, h_dim, n_classes], final_act=False)

    def forward(self, p4: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # The only scalar input is the particle's own invariant mass — anything
        # else (pT, η) would smuggle a frame in through the front door and quietly
        # destroy the equivariance the architecture is supposed to provide.
        m2 = psi(minkowski(p4, p4)).unsqueeze(-1)
        h = self.embed(m2) * mask.unsqueeze(-1)
        x = p4

        for blk in self.blocks:
            h, x = blk(h, x, mask)

        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(pooled)
