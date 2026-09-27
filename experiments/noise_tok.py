"""Supervised "generator noise word" features, cached per split as $ER_WORK/feat_noise_<map>/{split}.parquet.

A true copy's name may gain a word from a small noise set (center, services, partners, plus, inc, ...);
a sibling business gains a real descriptor (group, holdings, club, ecole, ...). For each name token t:
    noise(t) = #(t is an extra token of a true match) / (#S2/S3 train names containing t + 10)
counted on the ground truth of S1s outside trn/val (no leakage). Tokens never seen as noise score 0,
which is what makes French vocabulary swaps (club, comite, sportive, ...) look like siblings.
FR_EN maps a few French legal/generic words onto their English noise equivalent before lookup.
Pair features: nz_min / nz_max over the candidate's extra tokens (NaN when none), nz_zero = count of
extra tokens with noise 0.
usage: python noise_tok.py        (writes train + test)
"""
import os, sys
from collections import Counter
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/../business_entity_resolution/src")
import common as C, features as F  # noqa: E402

W = os.environ.get("ER_WORK", "/Users/archdex/Desktop/results/work2")
# $NOISE_MAP selects the French -> English lookup map and the output dir feat_noise_<map>:
#  v5    : generic words only (compagnie/societe -> company, groupe -> group, holding -> holdings, ...)
#  legal : v5 + French legal forms -> llc (house-number agreement of sarl/sas/eurl on test France is
#          0.83-0.85, like English legal forms)
MAPS = {"v5": {"compagnie": "company", "societe": "company", "associes": "associates", "et": "and",
               "fils": "sons", "groupe": "group", "holding": "holdings", "developpement": "development"}}
MAPS["legal"] = {**MAPS["v5"], **dict.fromkeys(["sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei"], "llc")}
MAP_NAME = os.environ.get("NOISE_MAP", "v5")
FR_EN = MAPS[MAP_NAME]
ACRO = os.environ.get("ACRO") == "1"
OUT = f"{W}/feat_noise_{MAP_NAME}" + ("_acro" if ACRO else "")
_G = {}


def _extra(bounds):
    lo, hi = bounds
    return [F.tok_diff(a, b)[0] for a, b in zip(_G["a"][lo:hi], _G["b"][lo:hi])]


def extras(a, b):
    _G.update(a=a, b=b)
    n = len(a)
    step = max(1, -(-n // (os.cpu_count() * 8)))
    with C.fork_pool(os.cpu_count()) as p:
        out = [e for part in p.map(_extra, [(i, min(i + step, n)) for i in range(0, n, step)]) for e in part]
    _G.clear()
    return out


def noise_table():
    rec = pd.read_parquet(f"{W}/train_rec.parquet", columns=["id", "nf", "src"]).set_index("id")
    roles = pd.read_parquet(f"{W}/roles.parquet")
    gt = pd.read_parquet(f"{W}/gt.parquet")
    gt = gt[gt.s1.isin(roles.s1[~roles.role.isin(["trn", "val"])])]
    ex = extras(rec.nf.reindex(gt.s1.values).values, rec.nf.reindex(gt.cand.values).values)
    pos = Counter(t for e in ex for t in e)
    df = Counter(t for s in rec.nf.values[rec.src.values != 1] for t in set(s.split()))
    return {t: c / (df.get(t, 0) + 10) for t, c in pos.items()}


def initials(s):
    return "".join(t[0] for t in s.split())


def pair_feats(split, noise):
    """ACRO=1 adds `acro`: the candidate's core name is the initials of the S1's full or core name (2+
    letters; train truth rate at the same house number 98.5 %), and drops acronym tokens from the noise
    lookup (an unseen acronym such as 'ea' or 'cmsb' would otherwise score 0 = descriptor)."""
    rec = pd.read_parquet(f"{W}/{split}_rec.parquet", columns=["id", "nf", "nc"]).set_index("id")
    c = pd.read_parquet(f"{W}/{split}_cands.parquet", columns=["s1", "cand"])
    a_nf, a_nc = rec.nf.reindex(c.s1.values).values, rec.nc.reindex(c.s1.values).values
    b_nc = rec.nc.reindex(c.cand.values).values
    ex = extras(a_nf, rec.nf.reindex(c.cand.values).values)
    zmin, zmax, zzero, acro = [], [], [], []
    for e, fa, ca, cb in zip(ex, a_nf, a_nc, b_nc):
        if ACRO:
            ini = {initials(fa), initials(ca)}
            cbj = cb.replace(" ", "")
            acro.append(float(len(cbj) >= 2 and cbj in ini))
            e = [t for t in e if t not in ini]
        z = [noise.get(FR_EN.get(t, t), 0.0) for t in e]
        zmin.append(min(z) if z else np.nan)
        zmax.append(max(z) if z else np.nan)
        zzero.append(sum(v == 0 for v in z))
    c["nz_min"], c["nz_max"], c["nz_zero"] = np.float32(zmin), np.float32(zmax), np.float32(zzero)
    if ACRO:
        c["acro"] = np.float32(acro)
    c.to_parquet(f"{OUT}/{split}.parquet")
    print(split, len(c), "acronym pairs", int(sum(acro)), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    noise = noise_table()
    top = sorted(noise.items(), key=lambda kv: -kv[1])[:30]
    print("noise words:", [(t, round(v, 2)) for t, v in top], flush=True)
    for s in ("train", "test"):
        pair_feats(s, noise)
