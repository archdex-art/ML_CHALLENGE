"""Compute pair features once for all train (trn+val) and test candidates and cache them as parquet chunks,
so matcher variants (experiments/v3.py) retrain in minutes without re-featurizing.
usage: python featcache.py train|test      (reads $ER_ROOT/results_v2/work, writes $ER_ROOT/results_v2/feat)
work/ must hold: train_rec, test_rec, roles, gt (v1 run) and train_cands, test_cands (v2 run).
"""
import os, sys
import pandas as pd
ROOT = os.environ.get("ER_ROOT", "/home/siddhartha/ML-Challenge-2026")
sys.path.insert(0, f"{ROOT}/business_entity_resolution/src")
from train import featurize

W, O = f"{ROOT}/results_v2/work", f"{ROOT}/results_v2/feat"
os.makedirs(O, exist_ok=True)
split = sys.argv[1]
c = pd.read_parquet(f"{W}/{split}_cands.parquet")
for i, f in enumerate(featurize(W, split, c, 50000 if split == "train" else 100000)):
    f.to_parquet(f"{O}/{split}_{i:03d}.parquet"); print(split, i, len(f), flush=True)
