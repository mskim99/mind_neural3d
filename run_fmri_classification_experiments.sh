#!/usr/bin/env bash
# MinD-3D++ E3 classification (C3 category classifier, C4 hard-negative) experiments.
# Same architecture/split/optimizer as E3-V050, with independent optional losses.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/jionkim/mind3d++/fmri_objaverse}"
VIS_ROOT="${VIS_ROOT:-/data/jionkim/mind3dpp_gt_views}"
GT_ROOT="${GT_ROOT:-/data/jionkim/mind3d++/GTmeshes/objaverse/glbs}"
E3_ROOT="${E3_ROOT:-/data/jionkim/mind_3d_output/fmri_dual_alignment}"
OUT_ROOT="${OUT_ROOT:-/data/jionkim/mind_3d_output/fmri_classification_experiments}"
RUNS="${RUNS:-C3_01 C3_03 C4_01 C4_02 C34}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
PATIENT="${PATIENT:-0}"
WARMSTART="${WARMSTART:-0}" # 1 = load E3V050 seed-matched best_objce for separate fine-tuning
OVERWRITE="${OVERWRITE:-0}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/train_fmri_clip_align_mind3dpp_cls_hard.py}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "$WARMSTART" == "1" ]]; then
  [[ "$EPOCHS" == "100" ]] && EPOCHS="40"
  [[ "$LR" == "1e-4" ]] && LR="2e-5"
fi
mkdir -p "$OUT_ROOT"

for seed in $SEEDS; do
  for run in $RUNS; do
    wcls="0"; whard="0"
    case "$run" in
      C0)    ;; # control: repeat E3-V050 objective in the NEW training script
      C3_01) wcls="0.1" ;;
      C3_03) wcls="0.3" ;;
      C4_01) whard="0.1" ;;
      C4_02) whard="0.2" ;;
      C34)   wcls="0.1"; whard="0.1" ;;
      *) echo "[ERROR] Unknown run: $run (C0 C3_01 C3_03 C4_01 C4_02 C34)" >&2; exit 2 ;;
    esac
    label="${run}_seed${seed}"
    extra=()
    if [[ "$WARMSTART" == "1" ]]; then
      label="${run}_ft_seed${seed}"
      checkpoint="$E3_ROOT/E3V050_seed${seed}/fmri_clip_aligner_best_objce.pt"
      [[ -f "$checkpoint" ]] || { echo "[ERROR] Missing E3 warm-start: $checkpoint" >&2; exit 3; }
      extra=(--init_dual_ckpt "$checkpoint")
    fi
    output="$OUT_ROOT/$label"
    if [[ -f "$output/summary.json" && "$OVERWRITE" != "1" ]]; then
      echo "[*] Skipping finished: $output (OVERWRITE=1 to rerun)"
      continue
    fi
    echo "=================================================================="
    echo "[*] $run seed=$seed | cls=$wcls hard=$whard warmstart=$WARMSTART"
    echo "[*] out=$output | GPU=$CUDA_VISIBLE_DEVICES"
    echo "=================================================================="
    python -u "$TRAIN_SCRIPT" \
      --data_path "$DATA_ROOT" --subjects sub-0001/npy_data \
      --gt_mesh_root "$GT_ROOT" \
      --visual_views_root "$VIS_ROOT" \
      --visual_cache_path "$E3_ROOT/gt_multiview_clip_vitb32_openai.pt" \
      --visual_batch_size 48 --visual_view_count 6 \
      --batch_nce_weight 1 --full_ce_weight 0 --category_ce_weight 0.2 \
      --visual_loss_weight 0.5 --visual_nce_weight 1 --visual_cos_weight 0.5 \
      --geometry_loss_weight 0 \
      --cls_ce_weight "$wcls" --cls_head_type linear \
      --cls_label_smoothing 0.05 --cls_dropout 0.3 \
      --hard_negative_weight "$whard" --hard_same_category_k 4 \
      --hard_cross_category_k 4 --hard_margin 0.02 \
      --checkpoint_metric both \
      --num_frames 6 --fmri_norm none \
      --frame_dim 256 --hidden_dim 1024 \
      --temporal_layers 1 --temporal_heads 4 --dropout 0.2 \
      --epochs "$EPOCHS" --patience "$PATIENT" \
      --lr "$LR" --weight_decay 1e-2 \
      --batch_size 16 --eval_batch_size 16 \
      --split_seed 0 --seed "$seed" \
      --device cuda:0 --out_dir "$output" \
      "${extra[@]}"
  done
done

echo "[*] Classification experiments finished: $OUT_ROOT"
