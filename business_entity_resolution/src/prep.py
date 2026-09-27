"""Stage 0: learn transliteration maps, normalize every record, assign S1 roles.

Writes to --work: maps.json, gt.parquet, roles.parquet, {train,test}_rec.parquet
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import common as C

COLS = ["nf", "nc", "dom", "af", "nj", "ns", "nl_name", "nl_addr", "hn"]


def load(data, split, src):
    d = C.read_tsv(f"{data}/{split}/{split}_source{src}.tsv")
    d["src"] = np.int8(src)
    return d


def normalize(d, workers):
    with C.fork_pool(workers) as p:
        out = p.map(C.normalize_row, zip(d.business_name, d.business_address, d.country), chunksize=5000)
    r = pd.DataFrame(out, columns=COLS)
    r.insert(0, "id", C.id_to_int(d.entity_id).values)
    r["src"] = d.src.values
    r["country"] = d.country.values
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing train/ and test/")
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-val", type=int, default=50_000)
    ap.add_argument("--n-trn", type=int, default=200_000)
    ap.add_argument("--n-ft", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    gt = C.read_tsv(f"{a.data}/train/train_ground_truth.tsv")
    s1_all = C.id_to_int(gt.source1_entity_id).values
    ex = gt.assign(cand=gt.matched_entity_ids.str.split(",")).explode("cand")
    ex = ex[ex.cand.fillna("") != ""]
    pairs = pd.DataFrame({"s1": C.id_to_int(ex.source1_entity_id).values, "cand": C.id_to_int(ex.cand).values})
    pairs.to_parquet(f"{a.work}/gt.parquet")
    print("gt pairs", len(pairs), flush=True)

    rng = np.random.default_rng(0)
    perm = rng.permutation(np.unique(s1_all))
    roles = pd.DataFrame({"s1": perm, "role": "unused"})
    roles.loc[: a.n_val - 1, "role"] = "val"
    roles.loc[a.n_val: a.n_val + a.n_trn - 1, "role"] = "trn"
    roles.loc[a.n_val + a.n_trn: a.n_val + a.n_trn + a.n_ft - 1, "role"] = "ft"
    roles.to_parquet(f"{a.work}/roles.parquet")

    # learn native-script -> Latin maps from true pairs (training data only)
    raw = {s: load(a.data, "train", s) for s in (1, 2, 3)}
    pool = pd.concat([raw[2], raw[3]], ignore_index=True)
    nl = pool.business_name.str.contains(C.NONLATIN) | pool.business_address.str.contains(C.NONLATIN)
    pool = pool[nl].assign(cand=lambda d: C.id_to_int(d.entity_id).values)
    s1 = raw[1].assign(s1=lambda d: C.id_to_int(d.entity_id).values)
    j = pairs.merge(pool, on="cand").merge(s1, on="s1", suffixes=("_c", "_s"))
    tok, comp = C.learn_maps(zip(j.business_name_s, j.business_address_s, j.business_name_c, j.business_address_c))
    json.dump({"tok": tok, "comp": comp}, open(f"{a.work}/maps.json", "w"), ensure_ascii=False)
    print(f"learned {len(tok)} name tokens, {len(comp)} address components from {len(j)} pairs", flush=True)
    C._MAPS.update(tok=tok, comp=comp)  # inherited by forked workers

    for split in ("train", "test"):
        parts = [normalize(raw[s] if split == "train" else load(a.data, split, s), a.workers) for s in (1, 2, 3)]
        rec = pd.concat(parts, ignore_index=True)
        rec.to_parquet(f"{a.work}/{split}_rec.parquet")
        print(split, len(rec), rec.groupby(["src", "country"]).size().to_dict(), flush=True)
        if split == "train":
            del raw


if __name__ == "__main__":
    main()
