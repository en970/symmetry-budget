"""Deep Sets — permutation invariant, nothing more. Backbone for M0 and M1.

M0 and M1 are the *same network*; they differ only in whether Lorentz transforms
are applied to the training data. That is the point: it isolates the effect of
symmetry knowledge entering through data rather than through architecture, with
no architectural difference to confound it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes: list[int], act=nn.GELU, final_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_act:
            layers.append(act())
    return nn.Sequential(*layers)


class DeepSets(nn.Module):
    def __init__(self, in_dim: int = 7, hidden: int = 128, latent: int = 128, n_classes: int = 2):
        super().__init__()
        self.phi = mlp([in_dim, hidden, hidden, latent])
        self.rho = mlp([latent, hidden, hidden, n_classes], final_act=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.phi(x) * mask.unsqueeze(-1)
        # Mean over real constituents, not over the padded length: otherwise the
        # pooled representation would encode how much padding a jet happens to
        # carry, which is an artefact of the file format.
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.rho(pooled)
