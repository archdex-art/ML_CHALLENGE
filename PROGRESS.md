# Progress log — Amazon ML Challenge 2026: Business Entity Resolution

Status as of **2026-09-27**. Read this first; the code README is
[business_entity_resolution/README.md](business_entity_resolution/README.md).

## Where we stand

| | |
|---|---|
| Best public leaderboard | **0.973444** — v3 "nocluster" model, decision b = -2.5 |
| Leaderboard top | ~0.99 |
| Submissions | max **5 per day** (behaves like a rolling 24 h window) |
| Final choice | **v4c** (v4 features + cluster features, a=1.4 b=-2.5): `~/Desktop/results/output_v4c/`; models in `~/Desktop/results/work_v4c/` (`predict.py --work` reproduces it; `train.py` default = v4c) |
| Candidate set | 8.5 pairs per S1 (organizers rank smaller candidate sets higher in the final review) |

Full leaderboard history: [results_summary/leaderboard.csv](results_summary/leaderboard.csv).

## v4 (2026-09-27): sibling-token + house-number features, distractor weighting
Code: `common.house_number`, `features.token_stats / tok_diff / hn_relation`, `experiments/v4.py`.
Local work dir: `~/Desktop/results/work2` (prep re-run with the `hn` column; candidates = v1 candidates
filtered to forward top-8 ∪ reverse rank-1 → 8.49/S1 test, 8.23/S1 train, val recall ceiling 0.9897).

Findings behind it:
1. **Sibling descriptors are token-identifiable.** Distractors are the S1 name + a descriptor at a nearby house
   number. Descriptors never appear as the extra token of a true match in train: private (14282 negatives /
   0 positives), group 7476/0, holdings 7443/0, enterprises, industries, ventures, overseas, infratech, exports,
   north/south/…/midtown/harbor/summit (≈1500/0 each). Noise words of true matches (inc, services, center, lp)
   are balanced. The old normalization *removed* several of them (`private`, `group`, `holdings`, `india`, …
   are in `LEGAL`, so the core name hides them).
2. **The discriminator is label-free, so it transfers to France.** For every token one name has and the other
   lacks: the rate at which the pair's house numbers agree, per country, over all candidate pairs of the split.
   Sibling words: <1–5 %; noise words: 40–85 %. On test France the same statistic singles out international,
   distribution, participations, holding (2–5 %) vs sarl/sas/eurl/sci (83 %).
3. **House number parsing.** First number of the first non-unit/floor/box component
   (`Fl 1, Hillsboro, OR, 4544 Cornell Rd` → 4544). Relation classes: equal / one digit dropped / within 50 /
   further / one missing / both missing. Train match rates: equal 88 %, near 6 %, far 8 %. True copies differ
   from S1 by per-record noise only (S2 and S3 alike: 89–90 % equal, 2.4 % near, 4 % digit drop).
4. **Test has ~1.9x more distractors per S1, nothing else changed.** Test S2+S3 per S1 = 5.5–5.8 vs 4.7 in
   train; the address-less rate per S1 is identical (0.14/0.17), and the predicted matches-per-S1 histogram on
   test equals the true histogram on val. Training with negatives whose candidate matches no S1 at all weighted
   x2–3 (`WNEG`) moves the prior toward test and does not cost validation.
5. Remaining val loss is recall: 5768 FN vs 662 FP (1781 FN outside the candidate set); FNs are mostly
   address-less copies of names shared by several S1s, and random alias names at the S1's exact address.

| run | features / weights | val F0.5 (best a,b) | val @ a=1.4 b=-2.5 | test pairs @ b=-2.5 |
|---|---|---|---|---|
| base (≈ v3 nocluster, local rebuild) | no new features | 0.98322 | 0.98046 | 5 699 735 |
| v4 | + 12 new features, no cluster features | 0.98596 | 0.98441 | 5 704 832 |
| v4w2 | v4, WNEG=2 | 0.98613 | 0.98321 | 5 678 813 |
| v4w3 | v4, WNEG=3 | 0.98629 | 0.98275 | 5 682 793 |
| v4c | v4 + cluster features | 0.98668 | 0.98513 | 5 738 431 |

Probe files (all PASS the validator incl. `--check-ids`): `~/Desktop/results/probes/<run>_b<b>/`.
Submission order (one change per probe, compare with 0.973444 = v3 nocluster b=-2.5):
1. `v4_b-2.5` — new features only, same b as the best submission.
2. `v4w2_b-1.5` — distractor weighting, milder calibration shift (weighting replaces part of b).
3. `v4w3_b-1.5` / `v4w3_b-0.5` — stronger weighting; tells whether the weight or b carries the shift.
4. `v4c_b-2.5` — cluster features back; they cost 0.003 on the LB in v3, siblings are now explicit.

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

## Research after v4c (2026-09-27): where the remaining loss is
All numbers on v4c; "calibrated" = a=1.4, b=-2.5 (the submitted rule).

| measurement | result | reading |
|---|---|---|
| val loss split at the submitted rule (F 0.98513) | removing all FPs +0.0015; adding all in-candidate FNs +0.0106; 1781 FNs outside candidates | on val, recall is ~85 % of the loss |
| in-candidate FNs by type (5581) | address-less 3157, same tokens 793, low name similarity 658, swap 447, extra 293 | address-less copies dominate |
| address-less exact-name candidates vs number of S1s sharing the name | 1 S1: 93 % match (model 0.88); 2: 43 % (model 0.10); 3: 30 %; ≥6: 3 % | ambiguous by construction beyond 1 owner; b=-2.5 gives up the 2–3-owner cases |
| blocking misses (1781) | 71 % address-less; exact core-name match only 24 %, and a core name is shared by ~110 S2/S3 records per S1 on average | a name blocking pass would explode the candidate set for little recall → **not worth it** |
| uncertain pairs (0.1 < p' < 0.9) per S1 | val 0.10–0.11; test US 0.19, India 0.15, **France 0.44** | test is 1.5–4x harder than val; France is the hot spot |
| model-expected F0.5 on test (Monte Carlo from its own probabilities; optimistic by ~0.002 on val) | France 0.960, US 0.981, India 0.983, all 0.979 | loss shares ≈ France 0.006, US 0.007, India 0.008 |
| France uncertain pairs | 23 % are one-word swaps (`ptits sportive sci` vs `ptits federation sci`); 42 % at a near house number | French names come from a small generic vocabulary (club, comite, amicale, sportive…) |
| same-address swaps, US/India train | the swapped-in word is almost always the generator's noise set (center 99 %, services 99 %, service, partners, plus); real descriptors (holdings, public, ventures) are 0 % | in France the swapped-in words are ordinary French words, so train gives no direct evidence; a copy-slot test was inconclusive |
| learning curve (trn S1s) | 100k → 200k: best val +0.0005, at b=-2.5 +0.0011 | more training entities still help (needs blocking for more S1s) |
| capacity (depth 10, eta 0.03) | 0.98670 vs 0.98668 | XGBoost tuning is exhausted |
| ensembles of v4 variants | ≤ +0.0001 | not worth a submission |

## Next steps, ranked by expected leaderboard gain per effort
1. **v4c + distractor weighting** (`--wneg 2`, cached features, ~12 min): corrects the 1.9x test distractor
   prior inside the model instead of via b; with v4 it cost nothing on val. Probe at b=-2.5 and b=-1.5.
2. **Per-country decision rule**: France has 4x the uncertain pairs. Probe France at a stricter b (-3.5) and
   at a looser one (-1.5), US/India at -2.5; only the leaderboard can tell which way France is miscalibrated.
3. **Cross-encoder feature** (Kaggle run): a multilingual pair model reads French descriptors semantically
   (groupe ≈ group, holding ≈ holdings, participations ≈ holdings); train it on hard pairs (same street,
   descriptor / swap / near number) from train. Largest expected gain for France.
4. **More training entities**: block 400k more S1s (the `unused` role) and double trn; learning curve suggests
   +0.0005–0.001 at the submitted rule.
5. **Address-less shared names**: model p is 0.10 at 43 % true rate for 2-owner names; a per-S1 rule
   "add the address-less exact-name record when this S1 is the only owner of that name among S1s whose
   candidate lists contain it" is a cheap post-processing probe.
6. **Final package**: fill `Documentation_template.md` (team name + members needed); zip
   `output_v4c/` + `code/business_entity_resolution/` + documentation.
7. Unsubmitted probes, ready on Siddhartha's machine ([experiments/probes.py](experiments/probes.py)):
   `results_probes/ensemble4_b-2.5` (mean logit of the 4 v3 models) and `nocluster_{US,India,France}_strict`
   (one country at b = -3.5, the rest at -2.5). Also `results_v3_nocluster/b-2.0`. Leaderboard scores not yet known.
