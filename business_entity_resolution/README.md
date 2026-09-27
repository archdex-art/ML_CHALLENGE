# Business Entity Resolution — ML Challenge 2026

Pipeline: normalize → fine-tuned multilingual-e5-small embeddings → per-country kNN blocking
(both directions) → pair features → two-stage XGBoost → one-owner rule + expected-F0.5 set selection.

Models: `intfloat/multilingual-e5-small` (MIT, 118M params, fine-tuned), XGBoost (Apache 2.0).
No external data or services are used; the only download is the pretrained e5 checkpoint.

## Layout

| file | stage | output (in `--work`) |
|---|---|---|
| `src/common.py` | normalization, learned transliteration, F0.5 metric, decision rule | — |
| `src/prep.py` | learn native-script→Latin maps from train pairs; normalize all records; S1 roles (ft/trn/val) | `maps.json`, `gt.parquet`, `roles.parquet`, `{train,test}_rec.parquet` |
| `src/finetune.py` | contrastive fine-tune of e5 on `ft` S1 pairs | `e5ft/` |
| `src/block.py` | embeddings + exact GPU kNN per country (S1→top-K pool, pool→top-R S1) | `{split}_cands.parquet`, `recall.json` |
| `src/features.py` | pair features + stage-2 context features | — |
| `src/train.py` | stage-1 OOF, stage-2, decision tuning, validation report | `stage1.json`, `stage2.json`, `model_cfg.json`, `val_errors.tsv` |
| `src/predict.py` | score test, write submission files | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| `src/ce.py` | cross-encoder pair scores (2 folds, one per GPU) — extra matcher feature, experimental | `ce_fold{0,1}.parquet` |

## Run (Kaggle, GPU T4 x2, Internet ON)

Upload the challenge `dataset/` folder (containing `train/` and `test/`) as a Kaggle Dataset and this
folder as another; set the two paths below. Use "Save Version → Save & Run All" so a 12 h run survives
a closed browser. Roughly: prep 30 min, fine-tune 15 min, block 40 min per split, train 45 min, predict 60 min.

```bash
pip install -q rapidfuzz anyascii
DATA=/kaggle/input/<data-dataset>/dataset      # folder with train/ and test/
CODE=/kaggle/input/<code-dataset>/business_entity_resolution
cd /kaggle/working
python $CODE/src/prep.py     --data $DATA --work work
python $CODE/src/finetune.py --work work
python $CODE/src/block.py    --work work --split train      # prints blocking recall
python $CODE/src/train.py    --work work                    # prints validation F0.5
python $CODE/src/block.py    --work work --split test
python $CODE/src/predict.py  --work work --out output
pip freeze > requirements_kaggle.txt
```

Then download `output/` and validate from the challenge's `student_resource/` folder:
`python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

Keep `--k/--r` identical for the train and test `block.py` runs: rank features use K+1/R+1 as "absent".

Every stage reads/writes only `--work`, so any stage can be re-run alone.

## Run locally (MacBook Apple Silicon / Linux / CPU) — `run_all.sh`

Needs Python 3.10+ (3.12 tested). On a Mac: `brew install libomp` first (XGBoost needs it).

```bash
cd business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh /path/to/dataset          # folder containing train/ and test/
```

The device is picked automatically: CUDA → Apple GPU (MPS) → CPU. Re-running skips finished stages, so an
interrupted run resumes. Defaults are sized for a 16 GB machine; override with env vars, e.g.
`N_TRN=150000 N_VAL=40000 CHUNK=50000 ./run_all.sh ...` on 32 GB+. Keep the machine awake
(`caffeinate -i ./run_all.sh ...`) and expect several hours for the full 12M-record test set.
Outputs land in `./output/`; validate with the challenge's `utils/validate_submission.py`.
