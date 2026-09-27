"""Retrain stage 1 + 2 from cached features (results_v2/feat) with some features dropped and optional
upweighting of sibling-pattern negatives; score test; write probe submissions.
usage: python v3.py <tag> <comma-separated features to drop> [sibling-negative weight]
"""
import glob, itertools, json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, "/home/siddhartha/ML-Challenge-2026/business_entity_resolution/src")
import common as C, features as F
from train import predict, PARAMS
from predict import write_lists

tag, DROP = sys.argv[1], [x for x in sys.argv[2].split(",") if x]
WNEG = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
ROOT = os.environ.get("ER_ROOT", "/home/siddhartha/ML-Challenge-2026")
W, O = f"{ROOT}/results_v2/work", f"{ROOT}/results_v2/feat"
OUT = f"{ROOT}/results_v3_{tag}"; os.makedirs(OUT, exist_ok=True)
load = lambda s: pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{O}/{s}_*.parquet"))], ignore_index=True)


def fit(X, y, Xv, yv, w, rounds=5000):
    d = xgb.QuantileDMatrix(X, label=y, weight=w)
    dv = xgb.QuantileDMatrix(Xv, label=yv, ref=d)
    return xgb.train(PARAMS, d, rounds, evals=[(dv, "v")], early_stopping_rounds=100, verbose_eval=False)


roles = pd.read_parquet(f"{W}/roles.parquet"); roles = roles[roles.role.isin(["trn", "val"])]
gt = pd.read_parquet(f"{W}/gt.parquet"); gt = gt[gt.s1.isin(roles.s1)].drop_duplicates()
f = load("train")
f["y"] = f.merge(gt.assign(y=1), on=["s1", "cand"], how="left").y.fillna(0).values
f["role"] = f.s1.map(roles.set_index("s1").role)
feats1 = [c for c in f.columns if c not in F.AUX + ["y", "role"] + DROP]
trn, val = f[f.role == "trn"].reset_index(drop=True), f[f.role == "val"].reset_index(drop=True)
# sibling pattern (candidate at another house number, similar name + address) is more common in test
sib = (trn.fnum_c != "") & (trn.fn_eq == 0) & (trn.bfn_in == 0) & (trn.c_tset >= 70) & (trn.a_tset >= 70)
w = np.where(sib & (trn.y == 0), WNEG, 1.0)
fold = trn.s1.map(pd.Series(np.random.default_rng(0).integers(0, 4, trn.s1.nunique()), index=trn.s1.unique())).values

oof, iters = np.zeros(len(trn), np.float32), []
for k in range(4):
    tr, te = fold != k, fold == k
    bst = fit(trn.loc[tr, feats1], trn.y[tr], trn.loc[te, feats1], trn.y[te], w[tr])
    oof[te] = predict(bst, trn.loc[te, feats1]); iters.append(bst.best_iteration + 1)
n1 = int(np.mean(iters) * 1.1)
m1 = xgb.train(PARAMS, xgb.QuantileDMatrix(trn[feats1], label=trn.y, weight=w), n1); m1.set_attr(best_iteration=str(n1 - 1))
trn2, val2 = F.stage2(trn, oof), F.stage2(val, predict(m1, val[feats1]))
feats2 = [c for c in trn2.columns if c not in F.AUX + ["y", "role"] + DROP]
tr = fold != 0
m2 = fit(trn2.loc[tr, feats2], trn2.y[tr], trn2.loc[~tr, feats2], trn2.y[~tr], w[tr])
m1.save_model(f"{OUT}/stage1.json"); m2.save_model(f"{OUT}/stage2.json")

ids = roles.s1[roles.role == "val"].values; truth = gt[gt.s1.isin(ids)]
sc = pd.DataFrame({"s1": val.s1, "cand": val.cand, "p": predict(m2, val2[feats2])})
grid = {(a, b): C.f05_per_entity(C.decide(sc, mode="expf", a=a, b=b), truth, ids).mean()
        for a, b in itertools.product([1, 1.4, 2], [-4, -3, -2, -1, -.5, 0])}
(a0, b0), best = max(grid.items(), key=lambda kv: kv[1])
json.dump({"drop": DROP, "wneg": WNEG, "feats1": feats1, "feats2": feats2, "val_F05": best, "a": a0, "b": b0,
           "grid": {str(k): v for k, v in grid.items()}}, open(f"{OUT}/cfg.json", "w"), indent=1)
print(tag, "val F05", round(best, 5), "a", a0, "b", b0, "| F05 at a=1.4:",
      {b: round(grid[(1.4, b)], 5) for b in (-4, -3, -2, -1, -.5, 0)}, flush=True)

scored = []
for p in sorted(glob.glob(f"{O}/test_*.parquet")):
    t = pd.read_parquet(p); t2 = F.stage2(t, predict(m1, t[feats1]))
    scored.append(pd.DataFrame({"s1": t.s1, "cand": t.cand, "p": predict(m2, t2[feats2])}))
scored = pd.concat(scored, ignore_index=True); scored.to_parquet(f"{OUT}/test_scored.parquet")
s1_ids = pd.read_parquet(f"{W}/test_rec.parquet", columns=["id", "src"]).query("src == 1").id.values
for b in (-0.5, -2.5, -3.5):  # a fixed at 1.4 so b is comparable with the v2 / nocluster leaderboard probes
    sel = C.decide(scored, mode="expf", a=1.4, b=b)
    d = f"{OUT}/b{b}"; os.makedirs(d, exist_ok=True)
    write_lists(sel, s1_ids, "matched_entity_ids", f"{d}/matching_results.tsv")
    print(tag, "test b", b, "pairs", len(sel), "empty", len(s1_ids) - sel.s1.nunique(), flush=True)
