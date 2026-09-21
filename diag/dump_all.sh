#!/usr/bin/env bash
# One GPU task: dump the test-set features from each checkpoint, then run the
# thermal diagnostics over them.
#
#   bash diag/dump_all.sh                       # the two runs we care about
#   bash diag/dump_all.sh whu_pe_indep          # or name them
#
# About 3 minutes and ~290 MB per checkpoint (6,405 query + 93,609 gallery
# features at 768-d float32).
#
# THE FEATURE FILES STAY IN /tmp AND DIE WITH THE TASK.  They used to be copied
# back, on the theory that the dump was the expensive part and later questions
# would then be free.  That theory assumed the analysis could run on the dev
# machine, and it cannot in practice -- `--device cpu` takes twenty minutes, so
# every re-analysis is a GPU task anyway and re-dumping adds three minutes to a
# submission that was going to happen regardless.  Against that: 290 MB per
# checkpoint on a drive that has already filled up mid-run once and cost two
# hours of training.
#
# Files already on the shared drive from before this change are still picked up
# by the analysis below, so feats_whu_recipe_hihr.npz goes on serving as the
# frozen baseline to compare against.
#
# Defaults: `whu_recipe_hihr` is the clean baseline -- no positional delta, no
# text loss -- so it is the honest subject for "what is wrong with thermal".
# `whu_text_lam50` is our best run, and the comparison says whether anything we
# did changed the thermal geometry at all.
set -euo pipefail

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(whu_recipe_hihr whu_text_lam50)
NPZ=()                                  # what the analysis at the end reads

PY=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
REPO=/mnt/cache/wanghanzhi/HSY/HiHR
DATA=/mnt/cache/wanghanzhi/Datasets
LOG=$REPO/dump_features_log.txt

cd "$REPO"
: > "$LOG"

for MODE in "${RUNS[@]}"; do
  CFG=$REPO/configs/hihr_$MODE.yml
  # Checkpoint name follows SOLVER.MAX_EPOCHS, and the initialisation follows the
  # config's own PRETRAIN_PATH.  Both were hardcoded to 60 epochs and to
  # cargo_base, which silently excluded the 120-epoch runs and would have handed
  # whu_twin the wrong starting weights.
  EPOCHS=$(grep -aE '^ *MAX_EPOCHS:' "$CFG" | head -1 | tr -dc '0-9')
  CKPT=$REPO/out_hihr_$MODE/transformer_${EPOCHS:-60}.pth
  TMP=/tmp/feats_$MODE.npz

  if [ ! -f "$CFG" ] || [ ! -f "$CKPT" ]; then
    echo "skip $MODE (missing config or checkpoint)" | tee -a "$LOG"
    continue
  fi

  EXTRA=()
  # Same content-based checks as reeval.sh.  PRETRAIN_PATH still has to be
  # loadable even though TEST.WEIGHT overwrites it: make_model reads it while
  # constructing, and the WHU configs' 'self' path rejects the raw CLIP archive.
  if grep -q "^ *PRETRAIN_CHOICE: *'self'" "$CFG"; then
    SELF=$(grep -aE "^ *PRETRAIN_PATH:" "$CFG" | head -1 | sed "s/.*'\([^']*\)'.*/\1/")
    EXTRA+=(MODEL.PRETRAIN_PATH "$REPO/${SELF#./}")
  else
    EXTRA+=(MODEL.PRETRAIN_PATH "$DATA/ViT-B-16.pt")
  fi
  if grep -q '^ *TEXT_ALIGN: *True' "$CFG"; then
    EXTRA+=(MODEL.TEXT_CLIP_PATH "$DATA/ViT-B-16.pt")
  fi

  echo "===== $MODE" | tee -a "$LOG"
  "$PY" -u diag/dump_features.py \
    --config_file "$CFG" --weight "$CKPT" --out "$TMP" \
    DATASETS.ROOT_DIR "$DATA" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"

  NPZ+=("$TMP")
  echo "kept on the worker at $TMP (not copied back)" | tee -a "$LOG"
done

# The analysis has to run in this task, because the features it reads go away
# with the worker.  It is vectorised onto the GPU and takes about a minute, so
# that was never worth a second submission anyway.
#
# Anything left on the shared drive from before the copy-back was dropped joins
# the comparison, minus whatever this run just re-dumped -- otherwise a mode
# named on the command line would be analysed twice, once stale.
for f in "$REPO"/feats_*.npz; do
  [ -f "$f" ] || continue
  dup=0
  for n in "${NPZ[@]:-}"; do
    [ -n "$n" ] && [ "$(basename "$n")" = "$(basename "$f")" ] && dup=1
  done
  [ "$dup" = 0 ] && NPZ+=("$f")
done

echo | tee -a "$LOG"
echo "===== analysis" | tee -a "$LOG"
if [ ${#NPZ[@]} -eq 0 ]; then
  echo "nothing to analyse: no checkpoint was dumped and the shared drive has" \
       "no feature files" | tee -a "$LOG"
else
  "$PY" -u diag/analyse_modality.py "${NPZ[@]}" 2>&1 | tee -a "$LOG"
fi

echo | tee -a "$LOG"
echo "done.  log -> $LOG" | tee -a "$LOG"
echo "the dumped features were NOT copied back; re-run this script to ask a" | tee -a "$LOG"
echo "new question of the same checkpoint (3 min of forward per run)." | tee -a "$LOG"
