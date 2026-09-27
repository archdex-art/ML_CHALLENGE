"""Cross-encoder pair score (fold k of 2): fine-tune multilingual-e5-small as a pair classifier on the
raw text of half the 'trn' S1s' candidate pairs, then score the other half, all 'val' pairs and all test pairs.
Two folds keep the stacked XGBoost honest: every trn pair is scored by the model that did not train on it.
Run fold 0 and fold 1 on different GPUs in parallel. Output: {out}/ce_fold{k}.parquet (s1, cand, split, ce).
"""
import argparse
import math

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import common as C


def texts(data, split):
    d = pd.concat([C.read_tsv(f"{data}/{split}/{split}_source{s}.tsv") for s in (1, 2, 3)], ignore_index=True)
    return pd.Series((d.business_name + " | " + d.business_address).values, index=C.id_to_int(d.entity_id).values)


def batches(tok, a, b, bs, maxlen):
    for i in range(0, len(a), bs):
        yield tok(list(a[i:i + bs]), list(b[i:i + bs]), truncation=True, max_length=maxlen,
                  padding=True, return_tensors="pt")


@torch.no_grad()
def score(model, tok, a, b, dev, maxlen):
    if len(a) == 0:
        return np.empty(0, np.float32)
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])  # length-sorted batches: little padding
    out = np.empty(len(a), np.float32)
    model.eval()
    for i, enc in enumerate(batches(tok, a[order], b[order], 1024, maxlen)):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out[order[i * 1024:(i + 1) * 1024]] = model(**enc.to(dev)).logits[:, 0].float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="folder with train_cands, test_cands, gt, roles (v2 run)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--fold", type=int, choices=[0, 1], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap pairs per stage")
    a = ap.parse_args()
    dev = C.device()
    lim = (lambda d: d.head(a.limit)) if a.limit else (lambda d: d)

    roles = pd.read_parquet(f"{a.work}/roles.parquet")
    gt = pd.read_parquet(f"{a.work}/gt.parquet").drop_duplicates()
    tr = pd.read_parquet(f"{a.work}/train_cands.parquet", columns=["s1", "cand"])
    tr["role"] = tr.s1.map(roles.set_index("s1").role)
    half = pd.Series(np.random.default_rng(1).integers(0, 2, tr.s1.nunique()), index=tr.s1.unique())
    tr["half"] = np.where(tr.role == "trn", tr.s1.map(half), -1)
    tr["y"] = tr.merge(gt.assign(y=1), on=["s1", "cand"], how="left").y.fillna(0).values
    txt = texts(a.data, "train")

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1).to(dev)

    fit = lim(tr[tr.half == a.fold].sample(frac=1, random_state=a.fold))
    xa, xb, y = txt.loc[fit.s1].values, txt.loc[fit.cand].values, torch.tensor(fit.y.values, dtype=torch.float32)
    steps = math.ceil(len(fit) / a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, steps // 20, steps)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    for i, enc in enumerate(batches(tok, xa, xb, a.bs, a.maxlen)):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            logit = model(**enc.to(dev)).logits[:, 0]
        loss = lossf(logit.float(), y[i * a.bs:(i + 1) * a.bs].to(dev))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        if i % 1000 == 0:
            print(f"fold {a.fold} step {i}/{steps} loss {loss.item():.4f}", flush=True)

    out = []
    ev = lim(tr[(tr.half == 1 - a.fold) | (tr.role == "val")])
    out.append(ev[["s1", "cand"]].assign(split="train", ce=score(model, tok, txt.loc[ev.s1].values,
                                                                 txt.loc[ev.cand].values, dev, a.maxlen)))
    print(f"fold {a.fold} scored train", len(ev), flush=True)
    del txt
    te = lim(pd.read_parquet(f"{a.work}/test_cands.parquet", columns=["s1", "cand"]))
    txt = texts(a.data, "test")
    out.append(te.assign(split="test", ce=score(model, tok, txt.loc[te.s1].values, txt.loc[te.cand].values,
                                                dev, a.maxlen)))
    pd.concat(out, ignore_index=True).to_parquet(f"{a.out}/ce_fold{a.fold}.parquet")
    print(f"fold {a.fold} done", flush=True)


if __name__ == "__main__":
    main()
