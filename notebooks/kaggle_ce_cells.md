# Kaggle notebook: cross-encoder scores (`src/ce.py`)

Settings: GPU T4 x2, **Internet ON** (downloads `intfloat/multilingual-e5-small`). Inputs: the code dataset
(`er_code_v3` folder, contains `ce.py`), `aml2026-er-data`, the **v1 notebook output** (`ml-challenge-2026`:
roles, gt) and the **v2 notebook output** (`ml-challenge-2026-v2`: train/test candidates).
Run with *Save Version → Save & Run All*. Expected 2–5 h (CPU tokenization on Kaggle's 4 cores is the bottleneck).

Output: `ce_fold0.parquet`, `ce_fold1.parquet` with columns `s1, cand, split, ce` (logit).
- fold k trains on half k of the `trn` S1s' candidate pairs, then scores the other half, all `val` pairs and all test pairs.
- Use: trn pair → score from the fold that did **not** train on it; val/test pair → mean of both folds.

```python
# 1 install, pre-download the model once, locate inputs
!pip install -q rapidfuzz anyascii
from huggingface_hub import snapshot_download
snapshot_download("intfloat/multilingual-e5-small")
import glob, os
CODE = glob.glob("/kaggle/input/**/er_code_v3/src/ce.py", recursive=True)[0].rsplit("/", 1)[0]
DATA = glob.glob("/kaggle/input/**/train/train_source1.tsv", recursive=True)[0].rsplit("/train/", 1)[0]
V1 = "/kaggle/input/notebooks/siddharthagopala/ml-challenge-2026/work"
V2 = [p.rsplit("/", 1)[0] for p in glob.glob("/kaggle/input/**/test_cands.parquet", recursive=True)
      if not p.startswith(V1)][0]
W = "/kaggle/working/work"; os.makedirs(W, exist_ok=True)
for f, src in [("roles.parquet", V1), ("gt.parquet", V1), ("train_cands.parquet", V2), ("test_cands.parquet", V2)]:
    if not os.path.exists(f"{W}/{f}"): os.symlink(f"{src}/{f}", f"{W}/{f}")
print(CODE, DATA, V2, os.path.getsize(f"{W}/test_cands.parquet") // 2**20, "MB", sep="\n")   # expect ~217 MB
```
```python
%%bash -s "$CODE" "$W" "$DATA"
# 2 train + score both folds, one per GPU, in parallel. Progress goes to ce0.log / ce1.log only.
cd $1
CUDA_VISIBLE_DEVICES=0 python ce.py --work $2 --data $3 --fold 0 --out /kaggle/working > /kaggle/working/ce0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python ce.py --work $2 --data $3 --fold 1 --out /kaggle/working > /kaggle/working/ce1.log 2>&1
wait
grep -E "step|scored|done|Error" /kaggle/working/ce0.log /kaggle/working/ce1.log | tail -20
```
```python
# 3 check outputs
!ls -lh /kaggle/working/ce_fold*.parquet
```
Known gap: progress is invisible while cell 2 runs. For future runs, stream it (e.g. `tail -f` the logs from a
background thread) so a stuck run can be spotted.
