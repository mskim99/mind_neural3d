#!/bin/bash
# V-9 (11-28, 5090 GPU 0): 구조 후보 정답이 범주를 빼고도 fMRI에서 읽히나 — ② VAE latent(v9_vae.py) → 읽기(v9_read.py), 학습 없음
#  준비: $W/prism_v9/{v9_feat.npz, v5_feat.npz} (로컬 v9_feat.py · V-5 특징을 올림)
# 사용 (절대 경로로): nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_v9.sh > $W/logs/v9.log 2>&1 < /dev/null &
GPU=0 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
cd $W/run; PY=$W/venv/bin/python; O=$W/prism_v9; OUT=${OUT:-$O}; mkdir -p $OUT   # 11-28 추가 읽기: OUT=$O/add V9B=… TARGETS=…
wait_gpu_free v9
if [ ! -f $O/v9_vae.npz ]; then echo "=== VAE $(date +%H:%M:%S)"
  HF_HOME=$W/cache/hf HF_HUB_OFFLINE=1 $W/venv_unclip/bin/python $PR/v9_vae.py --frames $TGT/frames --ann $DR/annotations --out $O/v9_vae.npz | grep -vE "^\[" ; fi
VAE=; [ -f $O/v9_vae.npz ] && VAE="--vae $O/v9_vae.npz" || echo "VAE 없음 → ② 건너뜀"
echo "=== 읽기 $(date +%H:%M:%S)"
$PY $PR/v9_read.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_full/ausvp/model_final.pt --v9 $O/v9_feat.npz --v5 $O/v5_feat.npz $VAE ${V9B:+--v9b $V9B} ${TARGETS:+--targets $TARGETS} \
  --subs ${SUBS_V9:-sub-0001,sub-0002,sub-0003,sub-0004,sub-0005,sub-0006,sub-0007,sub-0008} --out $OUT
echo "=== V9_DONE $(date +%H:%M:%S)"
