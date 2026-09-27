# Progress log — Amazon ML Challenge 2026: Business Entity Resolution

Status as of **2026-09-27**. Read this first; the code README is
[business_entity_resolution/README.md](business_entity_resolution/README.md).

## Where we stand

| | |
|---|---|
| Best public leaderboard | **0.973444** — v3 "nocluster" model, decision b = -2.5 |
| Leaderboard top | ~0.99 |
| Submissions | max **5 per day** (behaves like a rolling 24 h window) |
| Running now | cross-encoder scores on Kaggle ([notebooks/kaggle_ce_cells.md](notebooks/kaggle_ce_cells.md)) |
| Candidate set | 8.5 pairs per S1 (organizers rank smaller candidate sets higher in the final review) |

Full leaderboard history: [results_summary/leaderboard.csv](results_summary/leaderboard.csv).

## Task recap
For each Source-1 (S1) record, list all matching S2/S3 records. Score = macro F0.5 per S1 (precision counts
2x; a singleton scores 1.0 when predicted empty, 0.0 otherwise). Train: India + US. Test adds **France**.
Rules: no external data lookup (disqualification); final models must be MIT/Apache-2.0 and ≤ 8B params.

## Pipeline (current best)
1. **prep.py** — learn native-script → Latin token / address-component maps from train matches (+ anyascii);
   normalize names (honorifics, abbreviations, legal forms → `full` and `core` name, domain flag) and addresses
   (state codes, number signs, ordinals, joined/split house numbers, per-country abbreviations incl. France).
   Assign S1 roles: 50k `val`, 200k `trn`, 400k `ft`.
2. **finetune.py** — contrastive fine-tune of `intfloat/multilingual-e5-small` (MIT) on one pair per `ft` S1.
3. **block.py** — per country, exact GPU kNN: S1 → top-8 S2/S3, plus reverse pairs where the S1 is the record's
   rank-1 S1. In the train split the `ft` S1s are **removed from the S1 pool** (their records become orphans).
4. **features.py** — ~60 pair features: retrieval ranks/cosines, rapidfuzz name/address similarities,
   house-number agreement, IDF-weighted overlaps, record flags, name/address frequency counts, within-S1 context.
5. **train.py** — XGBoost stage 1 (4-fold OOF) + stage 2 (context over stage-1 probabilities).
   *Best model drops 4 cluster features* (see finding 4) — implemented in [experiments/v3.py](experiments/v3.py),
   **not yet ported into train.py**.
6. **decide** (common.py) — calibration p' = sigmoid(a·logit p + b) with a = 1.4, b = -2.5; one owner per
   S2/S3 record; per-S1 expected-F0.5 subset or empty.
7. **predict.py** — writes `matching_results.tsv` and `candidate_pairs.tsv`.

## Key findings (in order of discovery)

1. **Validation lied at first.** v1 validated at 0.9818 but scored 0.954. Test has 5.5–5.8 S2/S3 records per S1
   vs 4.7 in train, in every country (not just France).
2. **Blocking can be 5x smaller for free.** Forward top-8 + reverse rank-1 = 8.5 candidates/S1 (was 45.6);
   costs 0.0001 F0.5 on validation and speeds featurize/predict ~5x.
3. **Test wants a stricter threshold.** Leaderboard rose monotonically with stricter b on v2
   (0.954 → 0.962 → 0.967 → 0.970); for v3 the peak is at b ≈ -2.5 (-3.5 already lower).
4. **Root cause: "sibling businesses".** On test, 45% of S1s have a cluster of ≥2 candidates sharing a
   *different* house number (16.5% on validation). In train such clusters are usually true matches (the S1's
   number is the odd one out), in test they are mostly sibling businesses: S1 name + extra word
   ("Ventures", "Group", "Exports"…) at a nearby number, 2–3 records each. Dropping the features that encode
   "cluster ⇒ match" (`g_same_fn`, `g_same_fn_top`, `fn_support`, `fn_top`) gave +0.003 on the leaderboard
   while costing 0.001 on validation.
5. **Validation is not a leaderboard proxy.** A domain-classifier reweighting of validation to look like test did
   not reproduce the leaderboard ordering → concept shift, not just covariate shift. Only the leaderboard can
   rank variants; spend submissions on controlled A/B probes (same b, one change).
6. On validation, 60% of the remaining loss is missed matches, mostly **address-less records whose name is
   shared by several S1s** (54% of S1s share their core name with another S1) — largely ambiguous from text.

## Tried and did not help (leaderboard)
- Upweighting sibling-pattern negatives in training (x4): 0.97326 vs 0.973444.
- Dropping 9 more within-S1 context features: 0.972429.
- Prior-shift (EM) correction of probabilities: estimated test prior ≈ train prior, no change.

## Validation numbers
| model | val F0.5 | notes |
|---|---|---|
| v1 | 0.98182 | oracle 0.99916, blocking recall 0.9971 |
| v2 | 0.98388 | orphan-style world; oracle 0.9973, recall 0.9897 |
| v3 nocluster | 0.98288 | |
| v3 w2 / w4 / lesscontext | 0.98261 / 0.98262 / 0.98263 | |

## Where the artifacts live (not in git: data is the organizers', outputs are 100 MB–8 GB)
| what | where |
|---|---|
| challenge data | organizers' student resource; Kaggle dataset `siddharthagopala/aml2026-er-data` (private) |
| code versions | Kaggle dataset `aml2026-er-code` (`er_code`, `er_code_v2`, `er_code_v3`) |
| v1 run: normalized records, maps, roles, gt, fine-tuned e5, v1 candidates/models | Kaggle notebook output `siddharthagopala/ml-challenge-2026` |
| v2 run: 8.5/S1 train+test candidates, v2 models, test scores | Kaggle notebook output `siddharthagopala/ml-challenge-2026-v2` |
| cached features, v3 variants, all submission files | Siddhartha's machine: `results_v2/feat/`, `results_v3_*/b*/matching_results.tsv` |

## Reproduce the best submission (0.973444)
1. v1 notebook ([business_entity_resolution/kaggle_run.ipynb](business_entity_resolution/kaggle_run.ipynb)) → prep, fine-tune.
2. v2 notebook ([notebooks/kaggle_v2_cells.md](notebooks/kaggle_v2_cells.md)) → train candidates, pruned test candidates.
3. Put v1 `work/{train_rec,test_rec,roles,gt}.parquet` and v2 `work/{train,test}_cands.parquet` into
   `$ER_ROOT/results_v2/work/`, then:
   ```bash
   export ER_ROOT=/path/to/checkout          # folder that contains business_entity_resolution/
   python experiments/featcache.py train && python experiments/featcache.py test     # ~10 min, 12 cores
   python experiments/v3.py nocluster g_same_fn,g_same_fn_top,fn_support,fn_top        # ~25 min
   ```
   → `results_v3_nocluster/b-2.5/matching_results.tsv`.

## Next steps (can be split across teammates)
1. **Cross-encoder feature** (running): when `ce_fold0/1.parquet` arrive, merge as a feature into the cached
   features and retrain v3-nocluster; probe at b = -2.5.
2. **Synthetic sibling entities** for training: for trn S1s, create 2–3 records with S1 name + extra word and a
   nearby house number, labelled non-match; needs re-embedding (Kaggle GPU). Not started.
3. **Final package** (due with the last submission): port the v3 feature drop (+ CE if kept) into `train.py`;
   regenerate `candidate_pairs.tsv` (8.5/S1); fill `Documentation_template.md` (team name + members needed);
   zip `output/` + `code/business_entity_resolution/` + documentation.
4. Unsubmitted probes, ready on Siddhartha's machine ([experiments/probes.py](experiments/probes.py)):
   `results_probes/ensemble4_b-2.5` (mean logit of the 4 v3 models) and `nocluster_{US,India,France}_strict`
   (one country at b = -3.5, the rest at -2.5). Also `results_v3_nocluster/b-2.0`. Leaderboard scores not yet known.
