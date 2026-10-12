#!/usr/bin/env bash
# E3 Visual + optional coarse GT geometry ablations.
# Requires 6-view GT images for every train_list UID (train + validation).
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/jionkim/mind3d++/fmri_objaverse}"
GT_ROOT="${GT_ROOT:-/data/jionkim/mind3d++/GTmeshes/objaverse/glbs}"
VIS_ROOT="${VIS_ROOT:-/data/jionkim/mind3d++/GTmeshes/objaverse/gt_views}"
OUT_ROOT="${OUT_ROOT:-/data/jionkim/mind_3d_output/fmri_dual_alignment}"
RUNS="${RUNS:-E3V025 E3V050 E3VG025}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-100}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$(dirname "$0")/train_fmri_clip_align_mind3dpp_dual.py}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT_ROOT"
python "$(dirname "$0")/check_fmri_gt_multiview.py" \
    --data_path "$DATA_ROOT" --gt_views_root "$VIS_ROOT" --gt_mesh_root "$GT_ROOT"
if [[ ! -d "$VIS_ROOT" ]]; then
  echo "[ERROR] Missing GT visual view root: $VIS_ROOT" >&2
  echo "Render GT views for train_list.txt using the provided Blender renderer first." >&2
  exit 1
fi

for seed in $SEEDS; do
  for run in $RUNS; do
    case "$run" in
      E3V010) wvis="0.10"; wgeo="0" ;;
      E3V025) wvis="0.25"; wgeo="0" ;;
      E3V050) wvis="0.50"; wgeo="0" ;;
      E3VG025) wvis="0.25"; wgeo="0.20" ;;
      *) echo "Unknown run: $run. Options: E3V010 E3V025 E3V050 E3VG025" >&2; exit 2 ;;
    esac
    output="$OUT_ROOT/${run}_seed${seed}"
    if [[ -f "$output/summary.json" && "${OVERWRITE:-0}" != "1" ]]; then
      echo "[*] Skipping finished run: $output (set OVERWRITE=1 to overwrite)"
      continue
    fi
    echo "========================================================================"
    echo "[*] $run seed=$seed | TextNCE=1 CatCE=0.2 Visual=$wvis Geometry=$wgeo"
    echo "[*] out_dir=$output | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "========================================================================"
    python -u "$TRAIN_SCRIPT" \
      --data_path "$DATA_ROOT" \
      --subjects "sub-0001/npy_data" \
      --gt_mesh_root "$GT_ROOT" \
      --visual_views_root "$VIS_ROOT" \
      --visual_cache_path "$OUT_ROOT/gt_multiview_clip_vitb32_openai.pt" \
      --geometry_cache_path "$OUT_ROOT/gt_shape30.pt" \
      --visual_view_count 6 --visual_batch_size 48 \
      --batch_nce_weight 1.0 --full_ce_weight 0 \
      --category_ce_weight 0.2 \
      --visual_loss_weight "$wvis" \
      --visual_nce_weight 1.0 --visual_cos_weight 0.5 \
      --geometry_loss_weight "$wgeo" \
      --checkpoint_metric both \
      --num_frames 6 --fmri_norm none \
      --frame_dim 256 --hidden_dim 1024 \
      --temporal_layers 1 --temporal_heads 4 --dropout 0.2 \
      --epochs "$EPOCHS" --patience 0 \
      --batch_size 16 --eval_batch_size 16 \
      --lr 1e-4 --weight_decay 1e-2 \
      --split_seed 0 --seed "$seed" \
      --device cuda:0 \
      --out_dir "$output"
  done
done

echo "[*] E3 runs finished. Results: $OUT_ROOT"
