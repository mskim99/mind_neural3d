#!/usr/bin/env bash
# Run C1 score fusion and C2 category linear/MLP probe on existing E3 checkpoints.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/jionkim/mind3d++/fmri_objaverse}"
E3_ROOT="${E3_ROOT:-/data/jionkim/mind_3d_output/fmri_dual_alignment}"
OUT_ROOT="${OUT_ROOT:-/data/jionkim/mind_3d_output/fmri_frozen_classification}"
VIS_ROOT="${VIS_ROOT:-/data/jionkim/mind3dpp_gt_views}"
RUNS="${RUNS:-E3V050 E3VG025}"
SEEDS="${SEEDS:-0}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best_objce}" # best_objce or best_10way
TASKS="${TASKS:-fusion probe}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for seed in $SEEDS; do
  for run in $RUNS; do
    checkpoint="$E3_ROOT/${run}_seed${seed}/fmri_clip_aligner_${CHECKPOINT_KIND}.pt"
    [[ -f "$checkpoint" ]] || { echo "[ERROR] Checkpoint not found: $checkpoint" >&2; exit 2; }
    for task in $TASKS; do
      out="$OUT_ROOT/${run}_${CHECKPOINT_KIND}_seed${seed}/$task"
      mkdir -p "$out"
      echo "[*] C1/C2 task=$task run=$run seed=$seed checkpoint=$CHECKPOINT_KIND"
      python -u "$SCRIPT_DIR/evaluate_fmri_classification.py" \
        --task "$task" --checkpoint "$checkpoint" --data_path "$DATA_ROOT" \
        --device cuda:0 --batch_size 16 --num_workers 4 \
        --feature_cache "$OUT_ROOT/_feature_cache/${run}_${CHECKPOINT_KIND}_seed${seed}" \
        --out_dir "$out" \
        --visual_cache "$E3_ROOT/gt_multiview_clip_vitb32_openai.pt" \
        --visual_views_root "$VIS_ROOT" \
        --alphas 0,0.25,0.5,0.75,1 --score_normalize zscore \
        --calibration_fraction 0.3 --calibration_seed 7 \
        --probe_heads linear,mlp --probe_epochs 50 \
        --probe_lr 1e-3 --probe_weight_decay 1e-2 \
        --probe_batch_size 128 --probe_dropout 0.3 --probe_seed "$seed"
    done
  done
done

echo "[*] Frozen classification experiments finished: $OUT_ROOT"
