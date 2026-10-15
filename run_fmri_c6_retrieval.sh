#!/usr/bin/env bash
# Evaluate C6 fMRI checkpoints with matched semantic/visual candidate scoring.
# Never uses official test for retrieval tuning or selection.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
ROOT="${ROOT:-/data/jionkim/mind_3d_output/fmri_c6_ablation}"
DATA_ROOT="${DATA_ROOT:-/data/jionkim/mind3d++/fmri_objaverse}"
VIS="${VIS:-/data/jionkim/mind_3d_output/fmri_dual_alignment/gt_multiview_clip_vitb32_openai.pt}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${HERE}/evaluate_fmri_classification.py}"
RUNS="${RUNS:-C6B0 C6R_DROP04 C6N_SAMPLE C6RN_COMBINED}"
SEEDS="${SEEDS:-0 1 2}"
SELECTS="${SELECTS:-best_10way}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "[ERROR] evaluate_fmri_classification.py missing: $EVAL_SCRIPT" >&2; exit 1
fi
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    [[ -d "$DATA_ROOT" ]] || { echo "[ERROR] DATA_ROOT does not exist: $DATA_ROOT" >&2; exit 1; }
    [[ -f "$VIS" ]] || { echo "[ERROR] visual feature cache does not exist: $VIS" >&2; exit 1; }
fi
for seed in $SEEDS; do
    for run in $RUNS; do
        for sel in $SELECTS; do
            ckpt="$ROOT/${run}_seed${seed}/fmri_clip_aligner_${sel}.pt"
            dest="$ROOT/fusion_diagnostics/${run}_${sel}_seed${seed}"
            if [[ -f "$dest/fusion_sweep.csv" && "${OVERWRITE:-0}" != "1" ]]; then
                echo "[*] Reusing existing: $dest/fusion_sweep.csv"
                continue
            fi
            if [[ "${DRY_RUN:-0}" != "1" && ! -f "$ckpt" ]]; then
                echo "[ERROR] Missing checkpoint: $ckpt" >&2; exit 1
            fi
            cmd=( "$PYTHON" -u "$EVAL_SCRIPT" \
                --task fusion --checkpoint "$ckpt" --data_path "$DATA_ROOT" \
                --visual_cache "$VIS" --alphas "$ALPHAS" \
                --score_normalize zscore --calibration_fraction 0.3 \
                --calibration_seed 7 --device cuda:0 \
                --feature_cache "$ROOT/_retrieval_feature_cache/${run}_${sel}_seed${seed}" \
                --out_dir "$dest" )
            echo "[*] Evaluating $run seed=$seed selection=$sel alphas=$ALPHAS"
            if [[ "${DRY_RUN:-0}" == "1" ]]; then printf '%q ' "${cmd[@]}"; printf '\n';
            else "${cmd[@]}"; fi
        done
    done
done
printf '[*] C6 retrieval diagnostics complete: %s/fusion_diagnostics\n' "$ROOT"
