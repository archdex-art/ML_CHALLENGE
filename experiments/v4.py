"""Feature cache + matcher experiments on the 8.5/S1 candidate sets.

usage:
  python v4.py feat train|test                      # cache pair features -> $ER_WORK/feat/{split}_NNN.parquet
  python v4.py fit <tag> <drop,cols> [b ...]        # stage 1 + 2 on trn, report val, score test, write probes
env WNEG (default 1): weight of negatives whose candidate matches no S1 anywhere ("distractors": siblings,
orphans). Test has ~1.9x more distractors per S1 than train (5.8 vs 4.7 S2/S3 records per S1 with the same
address-less rate), so WNEG ~2 moves the training prior toward test.
$ER_WORK must hold train_rec, test_rec, roles, gt, train_cands, test_cands.
env XGB: JSON overrides of train.PARAMS, e.g. XGB='{"max_depth": 10, "eta": 0.03}'.
"""
import glob, itertools, json, os, sys
import numpy as np, pandas as pd, xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/../business_entity_resolution/src")
import common as C, features as F  # noqa: E402
from train import featurize, predict, PARAMS  # noqa: E402
from predict import write_lists  # noqa: E402
PARAMS.update(json.loads(os.environ.get("XGB", "{}")))

W = os.environ.get("ER_WORK", "/Users/archdex/Desktop/results/work2")
O = f"{W}/feat"


def load(split, cols=None):
    return pd.concat([pd.read_parquet(p, columns=cols) for p in sorted(glob.glob(f"{O}/{split}_*.parquet"))],
                     ignore_index=True)


def feat(split):
    os.makedirs(O, exist_ok=True)
    c = pd.read_parquet(f"{W}/{split}_cands.parquet")
    for i, f in enumerate(featurize(W, split, c, 50000 if split == "train" else 100000)):
        f.to_parquet(f"{O}/{split}_{i:03d}.parquet")
        print(split, i, len(f), flush=True)


def fit(X, y, Xv, yv, w=None, rounds=5000):
    d = xgb.QuantileDMatrix(X, label=y, weight=w)
    dv = xgb.QuantileDMatrix(Xv, label=yv, ref=d)
    return xgb.train(PARAMS, d, rounds, evals=[(dv, "v")], early_stopping_rounds=100, verbose_eval=False)


def run(tag, drop, bs):
    out = f"{W}/runs/{tag}"
    os.makedirs(out, exist_ok=True)
    roles = pd.read_parquet(f"{W}/roles.parquet")
    roles = roles[roles.role.isin(["trn", "val"])]
    gt_all = pd.read_parquet(f"{W}/gt.parquet")
    gt = gt_all[gt_all.s1.isin(roles.s1)].drop_duplicates()
    f = load("train")
    f["y"] = f.merge(gt.assign(y=1), on=["s1", "cand"], how="left").y.fillna(0).values
    f["role"] = f.s1.map(roles.set_index("s1").role)
    feats1 = [c for c in f.columns if c not in F.AUX + ["y", "role"] + drop]
    trn, val = f[f.role == "trn"].reset_index(drop=True), f[f.role == "val"].reset_index(drop=True)
    # distractor = candidate that matches no S1 anywhere in train (sibling / orphan record)
    wneg = float(os.environ.get("WNEG", 1))
    w = np.where((trn.y == 0) & ~trn.cand.isin(gt_all.cand), wneg, 1.0).astype(np.float32)
    del f, gt_all
    fold = trn.s1.map(pd.Series(np.random.default_rng(0).integers(0, 4, trn.s1.nunique()),
                                index=trn.s1.unique())).values
    oof, iters = np.zeros(len(trn), np.float32), []
    for k in range(4):
        tr, te = fold != k, fold == k
        bst = fit(trn.loc[tr, feats1], trn.y[tr], trn.loc[te, feats1], trn.y[te], w[tr])
        oof[te] = predict(bst, trn.loc[te, feats1])
        iters.append(bst.best_iteration + 1)
    n1 = int(np.mean(iters) * 1.1)
    m1 = xgb.train(PARAMS, xgb.QuantileDMatrix(trn[feats1], label=trn.y, weight=w), n1)
    m1.set_attr(best_iteration=str(n1 - 1))
    trn2, val2 = F.stage2(trn, oof), F.stage2(val, predict(m1, val[feats1]))
    feats2 = [c for c in trn2.columns if c not in F.AUX + ["y", "role"] + drop]
    tr = fold != 0
    m2 = fit(trn2.loc[tr, feats2], trn2.y[tr], trn2.loc[~tr, feats2], trn2.y[~tr], w[tr])
    m1.save_model(f"{out}/stage1.json")
    m2.save_model(f"{out}/stage2.json")

    ids = roles.s1[roles.role == "val"].values
    truth = gt[gt.s1.isin(ids)]
    sc = pd.DataFrame({"s1": val.s1, "cand": val.cand, "p": predict(m2, val2[feats2])})
    sc.to_parquet(f"{out}/val_scored.parquet")
    grid = {(a, b): C.f05_per_entity(C.decide(sc, mode="expf", a=a, b=b), truth, ids).mean()
            for a, b in itertools.product([1, 1.4, 2], [-4, -3, -2.5, -2, -1, -.5, 0])}
    (a0, b0), best = max(grid.items(), key=lambda kv: kv[1])
    imp = pd.Series(m2.get_score(importance_type="gain")).sort_values(ascending=False)
    json.dump({"drop": drop, "feats1": feats1, "feats2": feats2, "val_F05": best, "a": a0, "b": b0,
               "iters1": n1, "iters2": m2.best_iteration + 1,
               "grid": {str(k): v for k, v in grid.items()}, "imp2": imp.head(40).round(1).to_dict()},
              open(f"{out}/cfg.json", "w"), indent=1)
    print(tag, "val F05", round(best, 5), "a", a0, "b", b0, "| a=1.4:",
          {b: round(grid[(1.4, b)], 5) for b in (-4, -3, -2.5, -2, -1, -.5, 0)}, flush=True)
    print(imp.head(25).round(1).to_string(), flush=True)
    del trn, trn2, val, val2

    scored = []
    for p in sorted(glob.glob(f"{O}/test_*.parquet")):
        t = pd.read_parquet(p)
        t2 = F.stage2(t, predict(m1, t[feats1]))
        scored.append(pd.DataFrame({"s1": t.s1, "cand": t.cand, "p": predict(m2, t2[feats2])}))
    scored = pd.concat(scored, ignore_index=True)
    scored.to_parquet(f"{out}/test_scored.parquet")
    s1_ids = pd.read_parquet(f"{W}/test_rec.parquet", columns=["id", "src"]).query("src == 1").id.values
    for b in bs:
        sel = C.decide(scored, mode="expf", a=1.4, b=b)
        d = f"{out}/b{b}"
        os.makedirs(d, exist_ok=True)
        write_lists(sel, s1_ids, "matched_entity_ids", f"{d}/matching_results.tsv")
        print(tag, "test b", b, "pairs", len(sel), "empty", len(s1_ids) - sel.s1.nunique(), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "feat":
        feat(sys.argv[2])
    else:
        run(sys.argv[2], [x for x in sys.argv[3].split(",") if x], [float(x) for x in sys.argv[4:]] or [-2.5])
