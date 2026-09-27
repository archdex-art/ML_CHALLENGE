# Kaggle notebook: v2 run (reuses the v1 run's outputs)

Settings: GPU T4 x2, Internet ON. Inputs: the code dataset (`aml2026-er-code`, must contain the v2+ code),
`aml2026-er-data`, and the **v1 notebook output** (`ml-challenge-2026`). Run with *Save Version → Save & Run All*.
Takes about 1.5 h. Produced the `ml-challenge-2026-v2` notebook output.

```python
# 1 setup: reuse v1 prep + fine-tuned model
!pip install -q rapidfuzz anyascii
import glob, os
V1 = "/kaggle/input/notebooks/siddharthagopala/ml-challenge-2026/work"
CODE = [p for p in glob.glob("/kaggle/input/**/business_entity_resolution/src", recursive=True)
        if "keep-r" in open(p + "/block.py").read()][0]          # picks v2+ code
W = "/kaggle/working/work"; os.makedirs(W, exist_ok=True)
for f in ["train_rec.parquet", "test_rec.parquet", "roles.parquet", "gt.parquet", "maps.json", "e5ft"]:
    if not os.path.exists(f"{W}/{f}"): os.symlink(f"{V1}/{f}", f"{W}/{f}")
print(CODE, os.listdir(W))
```
```python
# 2 train blocking (~1 h): 'ft' S1s removed from the S1 pool, forward top-8 + reverse rank-1
%cd {CODE}
!python block.py --work {W} --split train 2>&1 | grep -vE "it/s|s/it"
```
```python
# 3 train + validation report (~15 min)
!python train.py --work {W} 2>&1 | grep -vE "it/s|s/it"
```
```python
# 4 test candidates: v1 blocking pruned to v2's rule (identical to block.py --k 8 on test, saves ~1 h)
import pandas as pd
c = pd.read_parquet(f"{V1}/test_cands.parquet")
c = c[(c.frank <= 8) | (c.rrank <= 1)].copy(); c["frank"] = c.frank.clip(upper=9)
c.to_parquet(f"{W}/test_cands.parquet"); print(len(c), round(len(c) / c.s1.nunique(), 2))
```
```python
# 5 predict + collect files
!python predict.py --work {W} --out /kaggle/working/output 2>&1 | grep -v scored
!cp {W}/model_cfg.json {W}/recall.json {W}/val_errors.tsv /kaggle/working/output/ && ls -lh /kaggle/working/output
```
Note: symlinked inputs (roles, gt, rec parquets, e5ft) are **not** saved in this notebook's output; take them
from the v1 notebook output.
