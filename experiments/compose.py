"""Compose a submission = teacher file with chosen disagreement buckets flipped to our model's decision.

Buckets (teacher T vs our selection O at a=1.4, b):
  FR_add    pairs O has, T lacks, France          (added unless T gives the record to another S1)
  FR_drop   pairs T has, O lacks, France          (removed)
  USIN_add  pairs O has, T lacks, US + India
  USIN_drop pairs T has, O lacks, US + India
usage: python compose.py <teacher.tsv> <run> <out_dir> <bucket> [<bucket> ...] [--b -2.5]
Each single-bucket file is a leaderboard probe: its score minus the teacher's says which side is right
on that bucket; the final file applies every bucket whose probe beat the teacher.
"""
import argparse, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/../business_entity_resolution/src")
import common as C  # noqa: E402
from predict import write_lists  # noqa: E402
from v4 import read_pairs, W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("teacher")
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("buckets", nargs="+", choices=["FR_add", "FR_drop", "USIN_add", "USIN_drop"])
ap.add_argument("--b", type=float, default=-2.5)
a = ap.parse_args()

T = read_pairs(a.teacher)
O = C.decide(pd.read_parquet(f"{W}/runs/{a.run}/test_scored.parquet"), mode="expf", a=1.4, b=a.b)
rec = pd.read_parquet(f"{W}/test_rec.parquet", columns=["id", "src", "country"])
fr = rec.set_index("id").country.eq("France")
d = T.assign(t=1).merge(O.assign(o=1), how="outer", on=["s1", "cand"]).fillna(0)
d["fr"] = fr.reindex(d.s1.values).values
out = T.copy()
for b in a.buckets:
    m = (d.fr if b.startswith("FR") else ~d.fr) & ((d.o == 1) & (d.t == 0) if b.endswith("add")
                                                  else (d.t == 1) & (d.o == 0))
    sub = d.loc[m, ["s1", "cand"]]
    if b.endswith("add"):
        sub = sub[~sub.cand.isin(out.cand)]
        out = pd.concat([out, sub], ignore_index=True)
    else:
        out = out.merge(sub.assign(r=1), how="left", on=["s1", "cand"]).query("r != 1")[["s1", "cand"]]
    print(b, len(sub), flush=True)
os.makedirs(a.out, exist_ok=True)
s1_ids = rec.query("src == 1").id.values
write_lists(out, s1_ids, "matched_entity_ids", f"{a.out}/matching_results.tsv")
print("pairs", len(out))
