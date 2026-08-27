"""The load-bearing test: M3 must be Lorentz invariant at the output.

If this fails, M3 is not the model the protocol claims it is, and every number
produced by the grid is measuring something else.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from src.data.symmetry import random_lorentz, apply
from src.models.lorentznet import LorentzNet


def make_jets(b=4, p=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    p3 = torch.randn(b, p, 3, generator=g)
    e = (p3**2).sum(-1, keepdim=True).sqrt() + 0.3      # timelike, E > |p|
    p4 = torch.cat([e, p3], dim=-1)
    mask = torch.ones(b, p, dtype=torch.bool)
    mask[:, -2:] = False                                 # exercise the padding path
    return p4 * mask.unsqueeze(-1), mask


def test_output_is_lorentz_invariant():
    torch.manual_seed(0)
    model = LorentzNet(h_dim=16, m_dim=16, n_blocks=2).double().eval()
    p4, mask = make_jets()
    p4 = p4.double()

    with torch.no_grad():
        out = model(p4, mask)
        L = random_lorentz(p4.shape[0], max_beta=0.5, dtype=torch.float64)
        out_boosted = model(apply(L, p4) * mask.unsqueeze(-1), mask)

    dev = (out - out_boosted).abs().max().item()
    print(f"  logit sapması (boost altında): {dev:.3e}")
    assert dev < 1e-8, f"M3 Lorentz invariant DEĞİL: sapma {dev:.3e}"


def test_permutation_invariant():
    torch.manual_seed(0)
    model = LorentzNet(h_dim=16, m_dim=16, n_blocks=2).double().eval()
    p4, mask = make_jets()
    perm = torch.randperm(p4.shape[1])
    with torch.no_grad():
        a = model(p4.double(), mask)
        b = model(p4[:, perm].double(), mask[:, perm])
    dev = (a - b).abs().max().item()
    print(f"  logit sapması (permütasyon altında): {dev:.3e}")
    assert dev < 1e-9, f"permütasyon invariant değil: {dev:.3e}"


if __name__ == "__main__":
    test_output_is_lorentz_invariant()
    test_permutation_invariant()
    print("her iki simetri testi de geçti")
