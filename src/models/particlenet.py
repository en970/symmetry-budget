"""ParticleNet-lite — EdgeConv on a dynamic k-NN graph. M2, the geometry anchor.

Not Lorentz equivariant. It knows about *locality* in (η, φ) but nothing about
boosts. It sits between Deep Sets and LorentzNet, and exists so that any gap
between M1 and M3 can be checked against a model that adds structure without
adding symmetry — otherwise "equivariance helped" and "more expressive model
helped" are indistinguishable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .deepsets import mlp


def knn_graph(coords: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k nearest neighbours in `coords`, padding excluded."""
    d = torch.cdist(coords, coords)                                  # (B, P, P)
    d = d.masked_fill(~mask.unsqueeze(1), float("inf"))              # unreachable padding
    d = d.masked_fill(~mask.unsqueeze(2), float("inf"))
    d.diagonal(dim1=1, dim2=2).fill_(float("inf"))                   # exclude self
    return d.topk(k, dim=-1, largest=False).indices                  # (B, P, k)


class EdgeConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int = 8):
        super().__init__()
        self.k = k
        self.net = mlp([2 * in_dim, out_dim, out_dim])
        self.shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, P, C = h.shape
        idx = knn_graph(coords, mask, self.k)                        # (B, P, k)
        nb = torch.gather(h.unsqueeze(2).expand(B, P, self.k, C), 1,
                          idx.unsqueeze(-1).expand(B, P, self.k, C))
        center = h.unsqueeze(2).expand_as(nb)
        e = self.net(torch.cat([center, nb - center], dim=-1))         # (B, P, k, out)
        return (e.mean(dim=2) + self.shortcut(h)) * mask.unsqueeze(-1)


class ParticleNetLite(nn.Module):
    def __init__(self, in_dim: int = 7, hidden: int = 96, k: int = 8, n_classes: int = 2):
        super().__init__()
        self.embed = mlp([in_dim, hidden, hidden])
        self.conv1 = EdgeConv(hidden, hidden, k)
        self.conv2 = EdgeConv(hidden, hidden, k)
        self.head = mlp([hidden, hidden, n_classes], final_act=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        coords = x[..., :2]                                          # (Δη, Δφ)
        h = self.embed(x) * mask.unsqueeze(-1)
        h = self.conv1(h, coords, mask)
        h = self.conv2(h, h[..., :2], mask)                          # dynamic: graph follows features
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(pooled)
