# ML_CHALLENGE_HALFWIT — Amazon ML Challenge 2026: Business Entity Resolution

Best public leaderboard so far: **0.973444** (macro F0.5). Status, findings and next steps: **[PROGRESS.md](PROGRESS.md)**.

| folder | contents |
|---|---|
| [business_entity_resolution/](business_entity_resolution/) | the pipeline (prep → fine-tune → blocking → features → XGBoost → decision → submission files), plus `src/ce.py` cross-encoder |
| [experiments/](experiments/) | `featcache.py` (cache pair features), `v3.py` (retrain matcher variants from the cache, write probe submissions) |
| [notebooks/](notebooks/) | Kaggle cells for the v2 run and the cross-encoder run |
| [results_summary/](results_summary/) | leaderboard history, validation configs/metrics per version |

Not in this repo: the challenge dataset (organizers' data), model weights, candidate/feature caches and submission
files (100 MB–8 GB each). PROGRESS.md says where each one lives.

Requirements: Python 3.10+, `pip install -r business_entity_resolution/requirements.txt`; GPU steps run on Kaggle
(T4 x2, Internet ON).
