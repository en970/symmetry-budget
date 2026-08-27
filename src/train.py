"""Train exactly one grid cell and write its result.

Called once per cell, by `cell-runner` or by Kaggle. Everything it needs comes
from the cell spec; nothing about the protocol is decided here. In particular it
never chooses a model size, an axis value or a stopping point — those are fixed
in PROTOCOL.md and expanded by tools/grid.py.

    python3 -m src.train --cell <id> [--data-dir DIR]
    python3 -m src.train --smoke          # synthetic data, no download, ~20s
"""
from __future__ import annotations

import argparse, json, pathlib, time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data.toptagging import features, mask_of
from .data import symmetry as sym
from .models import SPECS, INPUT, build, n_params

ROOT = pathlib.Path(__file__).resolve().parents[1]


def background_rejection(y_true: np.ndarray, score: np.ndarray, eff_s: float = 0.3) -> float:
    """1/ε_B at signal efficiency ε_S. The protocol's primary metric.

    Reported as inf when no background survives the threshold — that is a real
    (if lucky) outcome and must not be silently rewritten as a large finite
    number, which would sail through a mean.
    """
    sig, bkg = score[y_true == 1], score[y_true == 0]
    if len(sig) == 0 or len(bkg) == 0:
        return float("nan")
    thr = np.quantile(sig, 1.0 - eff_s)
    eps_b = float((bkg >= thr).mean())
    return float("inf") if eps_b == 0 else 1.0 / eps_b


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC; avoids a sklearn dependency inside the Kaggle image."""
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def measure_flops(model: nn.Module, sample, mask) -> int:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as f:
            model(sample, mask)
        return int(f.get_total_flops())
    except Exception:
        return 0                      # recorded as 0 = unmeasured, never guessed


def load_npz(path: str, n: int, seed: int):
    """Load the prepared benchmark and take the pre-registered training slice.

    The subsample is drawn with the cell's seed rather than taken from the head:
    the N axis must vary the amount of data, not which corner of the file it came
    from, or the small-N cells would all share one idiosyncratic prefix.
    """
    z = np.load(path if str(path).endswith(".npz") else f"{path}/toptagging.npz")
    tr_p4, tr_y = z["train_p4"], z["train_y"]
    if n < len(tr_y):
        idx = np.random.default_rng(seed).choice(len(tr_y), size=n, replace=False)
        tr_p4, tr_y = tr_p4[idx], tr_y[idx]
    return (torch.from_numpy(tr_p4.copy()), torch.from_numpy(tr_y.astype(np.int64)),
            torch.from_numpy(z["test_p4"].copy()), torch.from_numpy(z["test_y"].astype(np.int64)))


def synthetic(n: int, p: int = 24, seed: int = 0):
    """Separable toy jets for the smoke test. Not physics — pipeline exercise only."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    p3 = torch.randn(n, p, 3, generator=g) * (1.0 + 0.35 * y[:, None, None])
    e = (p3 ** 2).sum(-1, keepdim=True).sqrt() + 0.3
    p4 = torch.cat([e, p3], dim=-1)
    mask = torch.ones(n, p, dtype=torch.bool)
    mask[:, -4:] = False
    return p4 * mask.unsqueeze(-1), y


# Anything above this is a remote job. The laptop this project is developed on
# has 16 GB, and LorentzNet's memory grows with the square of the constituent
# count — a full-size cell run locally takes the machine down. Enforced here
# rather than remembered, because it was already forgotten once.
LOCAL_CELL_LIMIT = 2_000


def guard_local(cell: dict, data_dir: str | None, force: bool) -> None:
    on_kaggle = pathlib.Path("/kaggle").exists()
    if on_kaggle or data_dir is None or force:
        return
    if cell["n"] > LOCAL_CELL_LIMIT or cell["model"] == "M3":
        raise SystemExit(
            f"refusing to run {cell['model']} n={cell['n']:,} on real data locally.\n"
            f"Heavy cells belong on Kaggle: `python3 tools/submit.py --cell {cell['id']}`.\n"
            f"Locally, use --smoke for the synthetic pipeline check, or pass --force "
            f"if you have a reason and the memory to back it."
        )


def run(cell: dict, data_dir: str | None, epochs: int, device: str,
        force: bool = False) -> dict:
    guard_local(cell, data_dir, force)
    torch.manual_seed(cell["seed"])
    np.random.seed(cell["seed"])

    model_id = cell["model"]
    augment = SPECS[model_id]["augment"]
    view = INPUT[model_id]

    if data_dir is None:
        p4_tr, y_tr = synthetic(cell["n"], seed=cell["seed"])
        p4_te, y_te = synthetic(max(cell["n"] // 2, 256), seed=cell["seed"] + 1000)
    else:
        p4_tr, y_tr, p4_te, y_te = load_npz(data_dir, cell["n"], cell["seed"])

    if cell["break"] == "acceptance":
        p4_tr, p4_te = sym.break_acceptance(p4_tr), sym.break_acceptance(p4_te)
    elif cell["break"] == "axis":
        p4_tr, p4_te = sym.break_axis(p4_tr), sym.break_axis(p4_te)

    model = build(model_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    # LorentzNet materialises (B, P, P, m_dim) tensors, so its memory grows with
    # the square of the constituent count. At P=100 and B=128 that is ~340 MB per
    # intermediate, several per block — enough to exhaust a 16 GB card. The batch
    # size is therefore a property of the model, not a global constant.
    batch = 32 if INPUT[model_id] == "p4" else 128
    loader = DataLoader(TensorDataset(p4_tr, y_tr), batch_size=batch, shuffle=True,
                        drop_last=False)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for p4b, yb in loader:
            p4b, yb = p4b.to(device), yb.to(device)
            if augment:
                # M1's entire difference from M0: the group acts on the data.
                L = sym.random_lorentz(p4b.shape[0], device=device, dtype=p4b.dtype)
                p4b = sym.apply(L, p4b) * mask_of(p4b).unsqueeze(-1)
            mask = mask_of(p4b)
            x = p4b if view == "p4" else features(p4b)
            loss = lossf(model(x, mask), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    train_seconds = time.time() - t0

    model.eval()
    scores = []
    with torch.no_grad():
        eval_batch = 32 if view == "p4" else 256
        for i in range(0, len(p4_te), eval_batch):
            b = p4_te[i:i + eval_batch].to(device)
            m = mask_of(b)
            xb = b if view == "p4" else features(b)
            scores.append(torch.softmax(model(xb, m), -1)[:, 1].cpu())
    s = torch.cat(scores).numpy()
    y = y_te.numpy()

    probe = p4_te[:8].to(device)
    pm = mask_of(probe)
    flops = measure_flops(model, probe if view == "p4" else features(probe), pm)

    return {
        **{k: cell[k] for k in ("id", "phase", "model", "n", "seed", "break", "budget")},
        "rej_at_30": background_rejection(y, s),
        "auc": auc_score(y, s),
        "accuracy": float(((s > 0.5).astype(int) == y).mean()),
        "train_seconds": round(train_seconds, 2),
        "n_params": n_params(model),
        "flops_per_forward": flops // max(len(probe), 1),
        "status": "ok",
        "synthetic": data_dir is None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell")
    ap.add_argument("--data-dir")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true",
                    help="override the local-machine guard (you probably do not want this)")
    a = ap.parse_args()

    if a.smoke:
        cells = [dict(id=f"smoke{m}", phase=0, model=m, n=512, seed=0,
                      **{"break": "none"}, budget="params") for m in ("M0", "M1", "M2", "M3")]
        for c in cells:
            r = run(c, None, epochs=2, device=a.device)
            print(f"  {c['model']}  auc={r['auc']:.3f}  rej@30={r['rej_at_30']:>7.2f}  "
                  f"{r['train_seconds']:>6.1f}s  {r['n_params']:>7,}p  "
                  f"{r['flops_per_forward']:>10,} flop/fwd")
        return

    grid = json.loads((ROOT / "experiments/grid.json").read_text())
    cell = next(c for c in grid if c["id"] == a.cell)
    result = run(cell, a.data_dir, a.epochs, a.device, force=a.force)
    out = ROOT / "results" / f"{cell['id']}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
