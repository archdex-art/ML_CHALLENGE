"""Leaderboard probes from saved test scores (no training):
ensemble of the v3 variants, and per-country threshold shifts around the best LB file (nocluster, b=-2.5)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/siddhartha/ML-Challenge-2026/business_entity_resolution/src")
import common as C
from predict import write_lists

R = "/home/siddhartha/ML-Challenge-2026"
rec = pd.read_parquet(f"{R}/results_v2/work/test_rec.parquet", columns=["id", "src", "country"])
ids = rec.id[rec.src == 1].values
country = rec.set_index("id").country


def write(sel, name):
    d = f"{R}/results_probes/{name}"; os.makedirs(d, exist_ok=True)
    write_lists(sel, ids, "matched_entity_ids", f"{d}/matching_results.tsv")
    print(f"{name:28s} pairs {len(sel):8d} empty {len(ids) - sel.s1.nunique()}", flush=True)


def logit(p):
    p = p.clip(1e-7, 1 - 1e-7); return np.log(p / (1 - p))


# ensemble: mean logit of the four no-cluster models (same pair set, same order)
models = ["nocluster", "w2", "w4", "lesscontext"]
sc = [pd.read_parquet(f"{R}/results_v3_{m}/test_scored.parquet") for m in models]
assert all((s.s1.values == sc[0].s1.values).all() and (s.cand.values == sc[0].cand.values).all() for s in sc)
ens = sc[0][["s1", "cand"]].copy()
ens["p"] = 1 / (1 + np.exp(-np.mean([logit(s.p.values) for s in sc], axis=0)))
write(C.decide(ens, mode="expf", a=1.4, b=-2.5), "ensemble4_b-2.5")

# per-country: one country at b=-3.5, the rest at -2.5 (LB of all at -2.5 = 0.973, all at -3.5 = 0.971253)
base = sc[0]; cty = base.s1.map(country).values
for c in ("US", "India", "France"):
    b = np.where(cty == c, -3.5, -2.5)
    p = 1 / (1 + np.exp(-(1.4 * logit(base.p.values) + b)))  # calibrate here, then decide with identity
    write(C.decide(base.assign(p=p), mode="expf", a=1.0, b=0.0), f"nocluster_{c}_strict")
