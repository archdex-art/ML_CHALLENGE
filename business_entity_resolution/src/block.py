"""Stage 2: candidate generation by dense kNN, per country, in both directions.

forward : each query S1 -> top-K S2/S3 records
reverse : each S2/S3 record -> top-R S1 records (all S1, so the rank reflects real competition)
Kept: forward top-K plus reverse pairs with rrank <= keep-r (reverse ranks up to R are still used as features).
Train split: 'ft' S1 records are removed from the S1 pool, so their S2/S3 records become orphans, as in test
(test has ~19% more S2/S3 per S1 than train: entities whose S1 record is absent).
Output {split}_cands.parquet, with the retrieval features:
  cos, frank (K+1 if absent), rrank (R+1 if absent), s1_top (best fwd cos), c_top (best rev cos)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import common as C

DEV = C.device()
CUDA = DEV != "cpu"  # fp16 on cuda and mps


def encode(model, pool, texts):
    out = []
    for i in range(0, len(texts), 500_000):
        chunk = list(texts[i:i + 500_000])
        if pool:
            e = model.encode_multi_process(chunk, pool, batch_size=512, normalize_embeddings=True)
        else:
            e = model.encode(chunk, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        out.append(np.asarray(e, dtype=np.float16 if CUDA else np.float32))
    return np.concatenate(out)


def topk(q, db, k):
    """Exact inner-product top-k, chunked so the score matrix stays ~1 GB."""
    dev = DEV
    db = torch.from_numpy(db).to(dev)
    k = min(k, len(db))
    bs = max(1, int(5e8 // len(db)))
    if dev == "mps":
        bs = max(1, bs // 4)  # MPS buffers are capped; keep the score matrix small
    vs, ix = [], []
    for i in range(0, len(q), bs):
        v, j = (torch.from_numpy(q[i:i + bs]).to(dev) @ db.T).topk(k, dim=1)
        vs.append(v.float().cpu())
        ix.append(j.cpu())
    del db
    return torch.cat(vs).numpy(), torch.cat(ix).numpy(), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--model", default=None, help="default: {work}/e5ft")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--keep-r", type=int, default=1)
    a = ap.parse_args()

    rec = pd.read_parquet(f"{a.work}/{a.split}_rec.parquet", columns=["id", "src", "country", "nf", "af"])
    rec["text"] = C.PREFIX + rec.nf + " | " + rec.af
    if a.split == "train":
        roles = pd.read_parquet(f"{a.work}/roles.parquet")
        queries = set(roles.s1[roles.role.isin(["trn", "val"])])
        rec = rec[~rec.id.isin(roles.s1[roles.role == "ft"])]
    model = SentenceTransformer(a.model or f"{a.work}/e5ft", device=DEV)
    model.max_seq_length = 64
    if CUDA:
        model.half()
    pool = model.start_multi_process_pool() if DEV == "cuda" and torch.cuda.device_count() > 1 else None

    out = []
    for country, g in rec.groupby("country"):
        s1, cand = g[g.src == 1], g[g.src != 1]
        if len(s1) == 0 or len(cand) == 0:
            continue
        e1, ec = encode(model, pool, s1.text.values), encode(model, pool, cand.text.values)
        s1_ids, c_ids = s1.id.values, cand.id.values
        qi = np.arange(len(s1)) if a.split == "test" else np.flatnonzero(s1.id.isin(queries).values)

        fv, fi, k = topk(e1[qi], ec, a.k)
        fwd = pd.DataFrame({"s1": np.repeat(s1_ids[qi], k), "cand": c_ids[fi.ravel()], "cos": fv.ravel(),
                            "frank": np.tile(np.arange(1, k + 1, dtype=np.int16), len(qi))})
        rv, ri, r = topk(ec, e1, a.r)
        rev = pd.DataFrame({"s1": s1_ids[ri.ravel()], "cand": np.repeat(c_ids, r), "cos": rv.ravel(),
                            "rrank": np.tile(np.arange(1, r + 1, dtype=np.int16), len(c_ids))})
        rev = rev[rev.s1.isin(s1_ids[qi])]
        m = fwd.merge(rev, on=["s1", "cand"], how="outer", suffixes=("", "_r"))
        m["cos"] = m.cos.fillna(m.cos_r).astype(np.float32)
        m["frank"] = m.frank.fillna(a.k + 1).astype(np.int16)
        m["rrank"] = m.rrank.fillna(a.r + 1).astype(np.int16)
        m = m[(m.frank <= a.k) | (m.rrank <= a.keep_r)]
        m["s1_top"] = m.s1.map(pd.Series(fv[:, 0], index=s1_ids[qi])).astype(np.float32)
        m["c_top"] = m.cand.map(pd.Series(rv[:, 0], index=c_ids)).astype(np.float32)
        out.append(m.drop(columns="cos_r"))
        print(country, "s1", len(qi), "pool", len(cand), "pairs", len(m), flush=True)
        del e1, ec

    if pool:
        model.stop_multi_process_pool(pool)
    cands = pd.concat(out, ignore_index=True)
    cands.to_parquet(f"{a.work}/{a.split}_cands.parquet")
    print("pairs", len(cands), "per S1", round(len(cands) / cands.s1.nunique(), 1))

    if a.split == "train":  # blocking recall report
        gt = pd.read_parquet(f"{a.work}/gt.parquet")
        gt = gt[gt.s1.isin(queries)].merge(cands, on=["s1", "cand"], how="left")
        rep = {f"fwd@{k}": float((gt.frank <= k).mean()) for k in (1, 5, 10, 20, 30, 40, 60, 100) if k <= a.k}
        rep["union"] = float(gt.cos.notna().mean())
        print(json.dumps(rep, indent=1))
        json.dump(rep, open(f"{a.work}/recall.json", "w"))


if __name__ == "__main__":
    main()
