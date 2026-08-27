"""Kaggle kernel: fetch the Top Tagging Reference Dataset and cut it to size.

Runs once. Every training cell then attaches this kernel's output instead of
re-downloading 2 GB, which at 120 cells would be the dominant cost of the whole
project.

The protocol needs at most 300k training jets and a fixed 40k test set, not the
full 1.2M — so the artefact is a few hundred MB rather than gigabytes.
"""
import numpy as np, pandas as pd, urllib.request, pathlib, sys

BASE = "https://zenodo.org/records/2603256/files"
N_TRAIN, N_TEST, P = 300_000, 40_000, 200
COLS = [f"{v}_{i}" for i in range(P) for v in ("E", "PX", "PY", "PZ")]


def fetch(name):
    dst = pathlib.Path(f"/kaggle/tmp/{name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE}/{name}?download=1", dst)
    print(f"  {name}: {dst.stat().st_size / 1e6:.0f} MB", flush=True)
    return dst


def load(path, n):
    df = pd.read_hdf(path, key="table", stop=n)
    p4 = df[COLS].to_numpy(dtype=np.float32).reshape(-1, P, 4)
    y = df["is_signal_new"].to_numpy(dtype=np.int8)
    return p4, y


tr_p4, tr_y = load(fetch("train.h5"), N_TRAIN)
te_p4, te_y = load(fetch("test.h5"), N_TEST)

# The file is distributed shuffled, but that is a claim about someone else's
# code. Check it here rather than inherit it: a class-ordered prefix would make
# every small-N cell in the grid meaningless.
for name, y in (("train", tr_y), ("test", te_y)):
    frac = float(y.mean())
    print(f"  {name}: {len(y):,} jets, signal fraction {frac:.4f}", flush=True)
    if not 0.4 < frac < 0.6:
        sys.exit(f"FATAL: {name} prefix is not class-balanced ({frac:.3f}) — "
                 f"the file is not shuffled and a prefix is not a sample")

# Constituents are pT-ordered and mostly empty past ~100; trimming halves the
# artefact with no information loss for this benchmark.
occupancy = (np.abs(tr_p4).sum(-1) > 0).sum(-1)
p_keep = int(min(P, max(60, np.percentile(occupancy, 99.5))))
print(f"  constituents kept: {p_keep} (99.5th percentile of occupancy)", flush=True)

np.savez_compressed(
    "/kaggle/working/toptagging.npz",
    train_p4=tr_p4[:, :p_keep], train_y=tr_y,
    test_p4=te_p4[:, :p_keep], test_y=te_y,
)
size = pathlib.Path("/kaggle/working/toptagging.npz").stat().st_size
print(f"wrote toptagging.npz — {size / 1e6:.0f} MB", flush=True)
