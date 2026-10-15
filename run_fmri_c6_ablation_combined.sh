#!/usr/bin/env bash
# C6: one-factor-at-a-time temporal / normalization / regularization ablation.
# Source: train_fmri_clip_align_mind3dpp_dual.py E3-V050; evaluation always center-6.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/data/jionkim/mind3d++/fmri_objaverse}"
GT_ROOT="${GT_ROOT:-/data/jionkim/mind3d++/GTmeshes/objaverse/glbs}"
VIS_ROOT="${VIS_ROOT:-/data/jionkim/mind3dpp_gt_views}"
CACHE_ROOT="${CACHE_ROOT:-/data/jionkim/mind_3d_output/fmri_dual_alignment}"
OUT_ROOT="${OUT_ROOT:-/data/jionkim/mind_3d_output/fmri_c6_ablation}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${HERE}/train_fmri_clip_align_mind3dpp_c6.py}"
RUNS="${RUNS:-C6B0 C6T_CENTER C6T_UNIFORM C6N_SAMPLE C6N_FRAME C6R_DROP04 C6R_WD05}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-100}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "[ERROR] Training script missing: $TRAIN_SCRIPT" >&2
    exit 1
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if [[ -f "${HERE}/check_fmri_gt_multiview.py" ]]; then
        "$PYTHON" "${HERE}/check_fmri_gt_multiview.py" \
            --data_path "$DATA_ROOT" --gt_views_root "$VIS_ROOT" --gt_mesh_root "$GT_ROOT"
    fi
    if [[ ! -d "$VIS_ROOT" ]]; then
        echo "[ERROR] Missing visual GT images: $VIS_ROOT" >&2
        exit 1
    fi
fi

for seed in $SEEDS; do
    for run in $RUNS; do
        sampling="random"; norm="none"; dropout="0.2"; wd="0.01"
        case "$run" in
            C6B0) ;;
            C6T_CENTER)  sampling="center" ;;
            C6T_UNIFORM) sampling="uniform" ;;
            C6N_SAMPLE)  norm="sample_zscore" ;;
            C6N_FRAME)   norm="frame_zscore" ;;
            C6R_DROP04)  dropout="0.4" ;;
            C6RN_COMBINED) norm="sample_zscore"; dropout="0.4" ;;
            C6R_WD05)    wd="0.05" ;;
            *) echo "[ERROR] Unknown run: $run" >&2; exit 2 ;;
        esac

        out_dir="$OUT_ROOT/${run}_seed${seed}"
        if [[ -f "$out_dir/summary.json" && "${OVERWRITE:-0}" != "1" ]]; then
            echo "[*] Skip completed run $run seed=$seed: $out_dir"
            continue
        fi

        echo "========================================================================"
        echo "[*] C6 $run seed=$seed | train_sampling=$sampling eval_sampling=center"
        echo "[*] norm=$norm dropout=$dropout weight_decay=$wd (loss = E3-V050)"
        echo "[*] OUT: $out_dir | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "========================================================================"

        cmd=( "$PYTHON" -u "$TRAIN_SCRIPT"
            --data_path "$DATA_ROOT" --subjects "sub-0001/npy_data"
            --gt_mesh_root "$GT_ROOT"
            --visual_views_root "$VIS_ROOT"
            --visual_cache_path "$CACHE_ROOT/gt_multiview_clip_vitb32_openai.pt"
            --visual_view_count 6 --visual_batch_size 48
            --batch_nce_weight 1.0 --full_ce_weight 0 --category_ce_weight 0.2
            --visual_loss_weight 0.5 --visual_nce_weight 1.0 --visual_cos_weight 0.5
            --geometry_loss_weight 0
            --checkpoint_metric both
            --num_frames 6 --train_frame_sampling "$sampling" --fmri_norm "$norm"
            --frame_dim 256 --hidden_dim 1024
            --temporal_layers 1 --temporal_heads 4 --dropout "$dropout"
            --epochs "$EPOCHS" --patience 0
            --batch_size 16 --eval_batch_size 16
            --lr 1e-4 --weight_decay "$wd"
            --split_seed 0 --seed "$seed" --device cuda:0 --out_dir "$out_dir"
        )

        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            printf '%q ' "${cmd[@]}"; printf '\n'
        else
            "${cmd[@]}"
        fi
    done
done

printf '[*] C6 ablations finished: %s\n' "$OUT_ROOT"
