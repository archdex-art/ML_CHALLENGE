#!/usr/bin/env bash
# One-command end-to-end run for a local machine (macOS Apple Silicon / Linux / CPU).
#   ./run_all.sh /path/to/dataset            # folder that contains train/ and test/
# Optional env: WORK (default ./work), OUT (default ./output), plus the size knobs below.
set -euo pipefail
DATA="${1:?usage: ./run_all.sh /path/to/dataset  (folder containing train/ and test/)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$PWD/work}"; OUT="${OUT:-$PWD/output}"

# Size knobs. Defaults are sized for a 16 GB laptop; raise them if you have more RAM (32 GB: 2-3x).
N_VAL="${N_VAL:-20000}"    # S1 entities held out for validation / threshold tuning
N_TRN="${N_TRN:-60000}"    # S1 entities the matcher trains on
N_FT="${N_FT:-100000}"     # S1 entities (1 pair each) for the embedding fine-tune
CHUNK="${CHUNK:-20000}"    # S1 entities per feature-building chunk (lower = less RAM)
FT_BS="${FT_BS:-64}"     # fine-tune batch size: 64 for laptops, 256 on a 24 GB GPU (used in the Kaggle run)
TRAIN_ARGS="${TRAIN_ARGS:-}" # matcher variant, default v4; e.g. "--keep-cluster" (v4c), "--wneg 2", "--b -1.5"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES   # macOS: allow fork() after torch is loaded
export PYTORCH_ENABLE_MPS_FALLBACK=1             # macOS: fall back to CPU for any op MPS lacks
export TOKENIZERS_PARALLELISM=false
cd "$HERE/src"

stage() { # skip a finished stage so an interrupted run can simply be re-launched
  local marker="$1"; shift
  if [ -e "$WORK/$marker" ]; then echo ">> skip (exists: $marker)"; else echo ">> $*"; "$@"; fi
}

stage roles.parquet    python prep.py --data "$DATA" --work "$WORK" --n-val "$N_VAL" --n-trn "$N_TRN" --n-ft "$N_FT"
stage e5ft/config.json python finetune.py --work "$WORK" --bs "${FT_BS:-64}"
stage train_cands.parquet python block.py --work "$WORK" --split train
stage model_cfg.json   python train.py --work "$WORK" --chunk "$CHUNK" $TRAIN_ARGS
stage test_cands.parquet  python block.py --work "$WORK" --split test
python predict.py --work "$WORK" --out "$OUT" --chunk "$CHUNK"
echo "Done. Send back: $OUT/matching_results.tsv, $OUT/candidate_pairs.tsv, $WORK/model_cfg.json, $WORK/recall.json"
