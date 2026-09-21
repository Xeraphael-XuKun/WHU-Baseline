#!/usr/bin/env bash
# Stage 0 -- the two diagnostics, in one GPU task.
#
#   bash diag/run_stage0.sh                          # baseline checkpoint
#   bash diag/run_stage0.sh whu_pe_indep             # some other run as arm B
#
# 0.1 needs no GPU and is repeated here anyway: it costs two seconds, and
# having both tables land in one log means the text-side and image-side
# readings can never come from different versions of the sentence table.
#
# 0.2 is not run.  It was measured on 2026-08-07 from feats_whu_recipe_hihr
# (diag/analyse_modality.py sections [1], [2], [2b]) and re-running it would
# only reproduce those numbers.  They are quoted at the end of this script so
# the three sections read together.
set -euo pipefail

MODE=${1:-whu_recipe_hihr}

PY=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
REPO=/mnt/cache/wanghanzhi/HSY/HiHR
DATA=/mnt/cache/wanghanzhi/Datasets
CLIP=$DATA/ViT-B-16.pt
CFG=$REPO/configs/hihr_$MODE.yml

# The epoch is read from the config rather than assumed: the 120-epoch runs
# write transformer_120.pth, and a hardcoded 60 sent this looking for a file
# that was never written.
EPOCHS=$(grep -aE '^ *MAX_EPOCHS:' "$CFG" | head -1 | tr -dc '0-9')
CKPT=$REPO/out_hihr_$MODE/transformer_${EPOCHS:-60}.pth

[ -f "$CFG" ]  || { echo "missing config: $CFG" >&2; exit 1; }
[ -f "$CLIP" ] || { echo "missing CLIP release: $CLIP" >&2; exit 1; }
[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT" >&2
                    echo "train $MODE first, or pass another run name." >&2; exit 1; }

cd "$REPO"
LOG=$REPO/stage0_$MODE.txt

{
  echo "=== $(date '+%F %T')  Stage 0 on $MODE"
  echo "=== config : $CFG"
  echo "=== arm A  : $CLIP"
  echo "=== arm B  : $CKPT"
} > "$LOG"

"$PY" -u diag/prompt_probe.py --clip "$CLIP" 2>&1 | tee -a "$LOG"

"$PY" -u diag/modality_zeroshot.py \
  --config_file "$CFG" \
  --clip "$CLIP" \
  --weight "$CKPT" \
  --root "$DATA" \
  --per-modality 2000 2>&1 | tee -a "$LOG"

{
  echo
  echo "=============================================================================="
  echo "Stage 0.2  图像端模态间隙 -- 不重跑，引用 2026-08-07 的测量"
  echo "  来源 diag/analyse_modality.py 节 [1][2][2b], feats_whu_recipe_hihr, 测试集"
  echo "------------------------------------------------------------------------------"
  echo "  模态质心之间的距离   RGB<->IR 0.0061   RGB<->Th 0.0103   IR<->Th 0.0104"
  echo "                       (噪声地板 0.0012, 三对都在地板之上)"
  echo "  d_cross(RGB,Th)      1.125      同 ID 跨模态"
  echo "  d_inter (d_diffpid)  1.273      同模态跨 ID"
  echo "  -> d_cross < d_inter, 文档 0.2 的判据(模态压过身份)不成立"
  echo
  echo "  但把身份/瞬间/相机/姿态全按住, 只换光谱:"
  echo "     RGB<->IR 孪生 0.1044   RGB<->Th 孪生 0.1549   不同的人 0.1906"
  echo "  -> 热成像的纯光谱代价已走完到陌生人的 81%; 近红外为负(比同模态换一帧还近)"
  echo "  -> 准确的说法不是'模态压过身份', 是'热成像压过身份, 近红外完全没有'"
  echo "=============================================================================="
} | tee -a "$LOG"

echo
echo "log: $LOG"
