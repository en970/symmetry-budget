"""Lorentz group actions: used for augmentation, and for breaking the symmetry.

Deliberately one module. M1 trains on jets transformed by these operations; M3
is built so that these operations commute with its layers; Phase 2 breaks the
symmetry with the same machinery. If augmentation drew from a different group
than the architecture preserves, the central comparison would be meaningless —
keeping them in one file makes that mismatch hard to introduce by accident.

Convention: four-momenta are (E, px, py, pz); the metric is (+,-,-,-).
"""
from __future__ import annotations

import torch

EPS = 1e-8


def minkowski(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """<a, b> = a0 b0 - a·b. The only inner product LorentzNet is allowed to use."""
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def boost_matrix(beta: torch.Tensor) -> torch.Tensor:
    """Pure boost with velocity beta, shape (..., 3) with |beta| < 1 -> (..., 4, 4)."""
    b2 = (beta * beta).sum(-1, keepdim=True).clamp(max=1 - 1e-6)
    gamma = torch.rsqrt(1 - b2)                                    # (..., 1)
    # Clamp rather than add an epsilon: at float64 a fixed 1e-8 in the
    # denominator injects a relative error of ~3e-8, which then shows up as a
    # spurious equivariance violation in the tests.
    tiny = torch.finfo(beta.dtype).tiny
    n = beta / b2.sqrt().clamp(min=tiny)                           # unit direction

    g = gamma.squeeze(-1)
    bg = (gamma * b2.sqrt()).squeeze(-1)
    outer = n.unsqueeze(-1) * n.unsqueeze(-2)                      # (..., 3, 3)
    eye = torch.eye(3, device=beta.device, dtype=beta.dtype).expand_as(outer)

    top_left = g.unsqueeze(-1).unsqueeze(-1)                       # (..., 1, 1)
    top_right = -(bg.unsqueeze(-1) * n).unsqueeze(-2)              # (..., 1, 3)
    bot_left = top_right.transpose(-1, -2)                         # (..., 3, 1)
    bot_right = eye + (g - 1).unsqueeze(-1).unsqueeze(-1) * outer

    top = torch.cat([top_left, top_right], dim=-1)
    bot = torch.cat([bot_left, bot_right], dim=-1)
    return torch.cat([top, bot], dim=-2)


def rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Spatial rotation embedded in 4x4 (Rodrigues). axis (...,3), angle (...,)."""
    n = axis / axis.norm(dim=-1, keepdim=True).clamp(min=torch.finfo(axis.dtype).tiny)
    c, s = torch.cos(angle), torch.sin(angle)
    zero = torch.zeros_like(n[..., 0])
    K = torch.stack([
        torch.stack([zero, -n[..., 2], n[..., 1]], -1),
        torch.stack([n[..., 2], zero, -n[..., 0]], -1),
        torch.stack([-n[..., 1], n[..., 0], zero], -1),
    ], dim=-2)
    eye3 = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    R3 = eye3 + s[..., None, None] * K + (1 - c)[..., None, None] * (K @ K)

    L = torch.zeros(*R3.shape[:-2], 4, 4, device=axis.device, dtype=axis.dtype)
    L[..., 0, 0] = 1.0
    L[..., 1:, 1:] = R3
    return L


def apply(L: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
    """Apply (B,4,4) transforms to (B,P,4) four-momenta."""
    return torch.einsum("bij,bpj->bpi", L, p4)


def random_lorentz(batch: int, max_beta: float = 0.4, device=None, generator=None,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A random boost composed with a random rotation.

    max_beta is bounded rather than uniform on the whole group: the Lorentz group
    is non-compact, so "uniformly random" has no meaning and an unbounded boost
    pushes constituents to numerical extremes. The bound is a hyperparameter of
    the augmentation, and M1's ceiling depends on it — that is a real limitation
    of the augmentation approach, not a flaw in this implementation, and it is
    stated in the report rather than tuned away.
    """
    kw = dict(device=device, generator=generator, dtype=dtype)
    direction = torch.randn(batch, 3, **kw)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp(
        min=torch.finfo(dtype).tiny)
    speed = torch.rand(batch, 1, **kw) * max_beta
    B = boost_matrix(direction * speed)

    axis = torch.randn(batch, 3, **kw)
    angle = torch.rand(batch, **kw) * 2 * torch.pi
    R = rotation_matrix(axis, angle)
    return B @ R


# --- Phase 2: controlled symmetry breaking ------------------------------------

def break_acceptance(p4: torch.Tensor, eta_max: float = 2.0) -> torch.Tensor:
    """Zero out constituents outside a pseudorapidity acceptance.

    This is what a real detector does, and it is not Lorentz invariant: a boosted
    copy of an accepted jet is not itself accepted. H2 predicts the equivariant
    model suffers more from this than the augmented one.
    """
    from .toptagging import kinematics, mask_of

    _, eta, _, _ = kinematics(p4)
    keep = (eta.abs() < eta_max) & mask_of(p4)
    return p4 * keep.unsqueeze(-1)


def break_axis(p4: torch.Tensor, strength: float = 0.15) -> torch.Tensor:
    """Introduce a preferred spatial direction by scaling one momentum component.

    A crude but transparent violation: unlike an acceptance cut it removes no
    information, so it separates "symmetry is broken" from "data is missing".
    """
    scale = torch.ones(4, device=p4.device, dtype=p4.dtype)
    scale[3] = 1.0 + strength
    return p4 * scale
