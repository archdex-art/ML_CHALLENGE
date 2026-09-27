"""Stage 3: two-stage XGBoost matcher + decision-rule tuning on held-out S1 entities.

stage 1 : pair features -> p1 (out-of-fold on trn S1s, full model for val/test)
stage 2 : stage-1 features + group context over p1 -> p2
decision: calibration (a, b) + expected-F0.5 subset selection (or threshold), tuned on val
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

import common as C
import features as F

PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
              device="cuda" if torch.cuda.is_available() else "cpu", max_depth=8, eta=0.05,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=2, max_bin=256)


def chunks(c, n):
    """Yield slices of c (sorted by s1) holding whole S1 groups, n S1s at a time."""
    u = c.s1.values
    b = list(np.searchsorted(u, np.unique(u)[::n])) + [len(c)]
    for lo, hi in zip(b[:-1], b[1:]):
        yield c.iloc[lo:hi]


def featurize(work, split, cands, chunk):
    rec = pd.read_parquet(f"{work}/{split}_rec.parquet").set_index("id")
    if split == "train":  # same S1 world as block.py: 'ft' S1s are absent
        roles = pd.read_parquet(f"{work}/roles.parquet")
        rec = rec[~rec.index.isin(roles.s1[roles.role == "ft"])]
    F.add_freq(rec)
    idf = F.idf_tables(rec)
    cands = cands.sort_values("s1", kind="stable").reset_index(drop=True)
    for ch in chunks(cands, chunk):
        yield F.build(rec, ch, idf)


def fit(X, y, Xv, yv, rounds=5000):
    d = xgb.QuantileDMatrix(X, label=y)
    dv = xgb.QuantileDMatrix(Xv, label=yv, ref=d)
    return xgb.train(PARAMS, d, rounds, evals=[(dv, "v")], early_stopping_rounds=100, verbose_eval=500)


def predict(bst, X):
    return bst.predict(xgb.DMatrix(X), iteration_range=(0, bst.best_iteration + 1))


def report(name, sel, truth, ids, meta):
    f = C.f05_per_entity(sel, truth, ids)
    m = meta.reindex(f.index)
    out = {"F05": round(f.mean(), 5),
           **{f"F05_{k}": round(v, 5) for k, v in f.groupby(m.country).mean().items()},
           "F05_singleton": round(f[m.nt == 0].mean(), 5), "F05_nonsingleton": round(f[m.nt > 0].mean(), 5)}
    print(name, json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=50_000)
    a = ap.parse_args()
    W = a.work

    roles = pd.read_parquet(f"{W}/roles.parquet")
    roles = roles[roles.role.isin(["trn", "val"])]
    gt = pd.read_parquet(f"{W}/gt.parquet")
    gt = gt[gt.s1.isin(roles.s1)].drop_duplicates()
    cands = pd.read_parquet(f"{W}/train_cands.parquet")
    f = pd.concat(featurize(W, "train", cands, a.chunk), ignore_index=True)
    f["y"] = f.merge(gt.assign(y=1), on=["s1", "cand"], how="left").y.fillna(0).values
    f["role"] = f.s1.map(roles.set_index("s1").role)
    feats1 = [c for c in f.columns if c not in F.AUX + ["y", "role"]]
    print("pairs", len(f), "pos rate", round(f.y.mean(), 4), "features", len(feats1), flush=True)

    trn, val = f[f.role == "trn"].reset_index(drop=True), f[f.role == "val"].reset_index(drop=True)
    fold = pd.Series(np.random.default_rng(0).integers(0, a.folds, trn.s1.nunique()), index=trn.s1.unique())
    fold = trn.s1.map(fold).values

    # stage 1: out-of-fold predictions for trn, full model for val
    oof, iters = np.zeros(len(trn), np.float32), []
    for k in range(a.folds):
        tr, te = fold != k, fold == k
        bst = fit(trn.loc[tr, feats1], trn.y[tr], trn.loc[te, feats1], trn.y[te])
        oof[te] = predict(bst, trn.loc[te, feats1])
        iters.append(bst.best_iteration + 1)
    n1 = int(np.mean(iters) * 1.1)
    m1 = xgb.train(PARAMS, xgb.QuantileDMatrix(trn[feats1], label=trn.y), n1)
    m1.set_attr(best_iteration=str(n1 - 1))
    p1_val = predict(m1, val[feats1])

    # stage 2
    trn2, val2 = F.stage2(trn, oof), F.stage2(val, p1_val)
    feats2 = [c for c in trn2.columns if c not in F.AUX + ["y", "role"]]
    es = fold == 0
    m2 = fit(trn2.loc[~es, feats2], trn2.y[~es], trn2.loc[es, feats2], trn2.y[es])
    p2_val = predict(m2, val2[feats2])

    # evaluation on val S1s (singletons included: every val S1 appears, candidates or not)
    val_ids = roles.s1[roles.role == "val"].values
    truth = gt[gt.s1.isin(val_ids)]
    meta = pd.DataFrame({"s1": val_ids}).set_index("s1")
    rec1 = pd.read_parquet(f"{W}/train_rec.parquet", columns=["id", "country"]).set_index("id").country
    meta["country"] = rec1.reindex(meta.index).values
    meta["nt"] = truth.groupby("s1").size().reindex(meta.index).fillna(0)
    res = {"recall_ceiling": round(len(truth.merge(val[["s1", "cand"]])) / len(truth), 5)}
    res["oracle"] = report("oracle (perfect matcher on candidates)", val.loc[val.y == 1, ["s1", "cand"]],
                           truth, val_ids, meta)
    res["stage1_thr0.5"] = report("stage1 thr=0.5", val.loc[p1_val >= 0.5, ["s1", "cand"]], truth, val_ids, meta)

    scored = pd.DataFrame({"s1": val.s1, "cand": val.cand, "p": p2_val})
    best = (-1, None)
    for mode, prm in [("thr", dict(thr=t)) for t in np.arange(0.2, 0.96, 0.05)] + \
                     [("expf", dict(a=x, b=y)) for x, y in itertools.product([0.7, 1, 1.4, 2], [-1, -.5, 0, .5, 1])]:
        s = C.f05_per_entity(C.decide(scored, mode=mode, **prm), truth, val_ids).mean()
        if s > best[0]:
            best = (s, dict(mode=mode, **{k: float(v) for k, v in prm.items()}))
    res["decision"] = best[1]
    sel = C.decide(scored, **best[1])
    res["final"] = report(f"final {best[1]}", sel, truth, val_ids, meta)

    m1.save_model(f"{W}/stage1.json")
    m2.save_model(f"{W}/stage2.json")
    json.dump({"feats1": feats1, "feats2": feats2, **res}, open(f"{W}/model_cfg.json", "w"), indent=1)

    # error dump for analysis
    rec = pd.read_parquet(f"{W}/train_rec.parquet", columns=["id", "nf", "af"]).set_index("id")
    fp = sel.merge(truth.assign(t=1), how="left").query("t != 1").assign(kind="FP")
    fn = truth.merge(sel.assign(t=1), how="left").query("t != 1").assign(kind="FN")
    err = pd.concat([fp.head(500), fn.head(500)])[["kind", "s1", "cand"]].merge(scored, how="left")
    for side in ("s1", "cand"):
        err[f"{side}_name"] = rec.nf.reindex(err[side]).values
        err[f"{side}_addr"] = rec.af.reindex(err[side]).values
    err.to_csv(f"{W}/val_errors.tsv", sep="\t", index=False)
    imp = pd.Series(m2.get_score(importance_type="gain")).sort_values(ascending=False)
    print("top stage-2 features:\n", imp.head(25).round(1).to_string())


if __name__ == "__main__":
    main()
