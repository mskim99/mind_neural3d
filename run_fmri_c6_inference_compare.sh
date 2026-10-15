#!/usr/bin/env bash
# C6 retrieval policies with exactly the same frozen checkpoint, UID split,
# candidate bank, SDXL settings and seed; no GT test data for caption retrieval.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-$HERE/inference_fmri_mind3dpp_c6_fusion.py}"
ROOT="${ROOT:-/data/jionkim/mind_3d_output/fmri_c6_ablation}"
DATA_ROOT="${DATA_ROOT:-/data/jionkim/mind3d_objaverse}"
VIS="${VIS:-/data/jionkim/mind_3d_output/fmri_dual_alignment/gt_multiview_clip_vitb32_openai.pt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/c6_inference_comparison}"
RUNS="${RUNS:-G0 G1 G2 G3}"
SEED="${SEED:-0}"
SELECT="${SELECT:-best_10way}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
SEMANTIC_ONLY="${SEMANTIC_ONLY:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCHSIZE="${BATCHSIZE:-8}"
MODEL_SEED="${MODEL_SEED:-42}" # SDXL/Zero123++ generation seed, not encoder training seed
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

[[ -f "$SCRIPT" ]] || { echo "[ERROR] script not found: $SCRIPT" >&2; exit 1; }
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -d "$DATA_ROOT" ]] || { echo "[ERROR] fMRI data not found: $DATA_ROOT" >&2; exit 1; }
fi
for run in $RUNS; do
    case "$run" in
        G0) encoder="C6R_DROP04"; alpha="1.0" ;;
        G1) encoder="C6RN_COMBINED"; alpha="1.0" ;;
        G2) encoder="C6RN_COMBINED"; alpha="0.25" ;;
        G3) encoder="C6RN_COMBINED"; alpha="0.5" ;;
        *) echo "[ERROR] Unknown run: $run (G0,G1,G2,G3)" >&2; exit 1 ;;
    esac
    ckpt="$ROOT/${encoder}_seed${SEED}/fmri_clip_aligner_${SELECT}.pt"
    out="$OUT_ROOT/${run}_${encoder}_${SELECT}_seed${SEED}_${EVAL_SPLIT}"
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        [[ -f "$ckpt" ]] || { echo "[ERROR] checkpoint not found: $ckpt" >&2; exit 1; }
        if [[ "$alpha" != 1.0 ]]; then
            [[ -f "$VIS" ]] || { echo "[ERROR] visual cache missing: $VIS" >&2; exit 1; }
        fi
    fi
    cmd=( "$PYTHON" -u "$SCRIPT" \
          --semantic_ckpt "$ckpt" \
          --data_path "$DATA_ROOT" \
          --eval_split "$EVAL_SPLIT" \
          --prompt_source train_caption \
          --fusion_alpha "$alpha" \
          --score_normalize zscore \
          --batchsize "$BATCHSIZE" \
          --num_workers "$NUM_WORKERS" \
          --seed "$MODEL_SEED" \
          --device cuda:0 \
          --out_dir "$out" )
    if [[ "$alpha" != 1.0 ]]; then cmd+=( --visual_cache "$VIS" ); fi
    if [[ "$SEMANTIC_ONLY" == 1 ]]; then cmd+=( --semantic_only ); fi
    if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=( --max_samples "$MAX_SAMPLES" ); fi
    if [[ "${CPU_OFFLOAD:-0}" == 1 ]]; then cmd+=( --cpu_offload ); fi
    # Optional for limited VRAM: move Zero123++ between CPU and GPU per sample.
    if [[ "${ZERO123_CPU_SWAP:-0}" == 1 ]]; then cmd+=( --zero123_cpu_swap ); fi
    echo "[*] $run | encoder=$encoder seed=$SEED checkpoint=$SELECT split=$EVAL_SPLIT alpha=$alpha semantic_only=$SEMANTIC_ONLY"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n';
    else "${cmd[@]}"; fi
done
