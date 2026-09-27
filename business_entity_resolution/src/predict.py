"""Stage 4: score test candidates and write output/matching_results.tsv + output/candidate_pairs.tsv."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import common as C
import features as F
from train import featurize, predict


def write_lists(pairs, s1_ids, col, path):
    s = pairs.assign(c=C.int_to_id(pairs.cand.values)).groupby("s1").c.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": C.int_to_id(s1_ids), col: s.reindex(s1_ids).fillna("").values})
    out.to_csv(path, sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--chunk", type=int, default=50_000)
    a = ap.parse_args()
    W = a.work
    cfg = json.load(open(f"{W}/model_cfg.json"))
    m1, m2 = xgb.Booster(model_file=f"{W}/stage1.json"), xgb.Booster(model_file=f"{W}/stage2.json")

    cands = pd.read_parquet(f"{W}/test_cands.parquet")
    scored = []
    for f in featurize(W, "test", cands, a.chunk):
        p1 = predict(m1, f[cfg["feats1"]])
        f2 = F.stage2(f, p1)
        scored.append(pd.DataFrame({"s1": f.s1, "cand": f.cand, "p": predict(m2, f2[cfg["feats2"]])}))
        print("scored", sum(map(len, scored)), "/", len(cands), flush=True)
    scored = pd.concat(scored, ignore_index=True)
    sel = C.decide(scored, **cfg["decision"])

    s1_ids = pd.read_parquet(f"{W}/test_rec.parquet", columns=["id", "src"]).query("src == 1").id.values
    os.makedirs(a.out, exist_ok=True)
    scored.to_parquet(f"{W}/test_scored.parquet")
    write_lists(sel, s1_ids, "matched_entity_ids", f"{a.out}/matching_results.tsv")
    write_lists(cands, s1_ids, "candidate_entity_ids", f"{a.out}/candidate_pairs.tsv")
    print("S1", len(s1_ids), "matched pairs", len(sel), "empty S1", len(s1_ids) - sel.s1.nunique(),
          "candidate pairs", len(cands))


if __name__ == "__main__":
    main()
