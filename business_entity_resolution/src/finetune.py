"""Stage 1: contrastive fine-tune of multilingual-e5-small on (S1, matched record) pairs.

Uses only S1 entities with role 'ft' (disjoint from the matcher's train/val S1s, so
retrieval cosine is not inflated on the pairs the matcher learns from).
"""
import argparse

import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()

    rec = pd.read_parquet(f"{a.work}/train_rec.parquet", columns=["id", "nf", "af"]).set_index("id")
    text = C.PREFIX + rec.nf + " | " + rec.af
    roles = pd.read_parquet(f"{a.work}/roles.parquet")
    gt = pd.read_parquet(f"{a.work}/gt.parquet")
    gt = gt[gt.s1.isin(roles.s1[roles.role == "ft"])]
    # one pair per S1 so in-batch negatives are never true matches of the same S1
    gt = gt.sample(frac=1, random_state=0).drop_duplicates("s1")
    ex = [InputExample(texts=[x, y]) for x, y in zip(text.loc[gt.s1].values, text.loc[gt.cand].values)]
    print("pairs", len(ex), flush=True)

    model = SentenceTransformer(a.base)
    model.max_seq_length = 64
    dl = DataLoader(ex, shuffle=True, batch_size=a.bs, drop_last=True)
    loss = losses.MultipleNegativesSymmetricRankingLoss(model)
    model.fit(train_objectives=[(dl, loss)], epochs=a.epochs, warmup_steps=len(dl) // 10,
              use_amp=torch.cuda.is_available(), show_progress_bar=True)
    model.save(f"{a.work}/e5ft")


if __name__ == "__main__":
    main()
