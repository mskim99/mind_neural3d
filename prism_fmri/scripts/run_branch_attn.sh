#!/bin/bash
# 새 구조 attention 진단 (11-15): branch 체크포인트는 branch_attn.py, 이전 구조(cat)는 dec_attn.py. 7명 × test 104, 토큰만
#  GPU=0 MODELS="aup ausvp40" bash run_branch_attn.sh   → $W/prism_attn/<이름>/
#  이름: aup(2만) · aup40 · au40 · ausvp40 (UKB 4안 + 새 구조) / vl4(UKB 4안 + 이전 구조, 4만) · g0(기존 Ours, 공개 인코더 + 이전 구조)
# 사용: GPU=0 MODELS="aup ausvp40" nohup bash run_branch_attn.sh > $W/logs/attn_g0.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
PY=$W/venv/bin/python; O=$W/prism_attn; cd $W/run; mkdir -p $O
for m in ${MODELS:-aup}; do
  case $m in
    aup) CK=$W/prism_branch/aup/model_final.pt;; aup40) CK=$W/prism_branch40/aup/model_final.pt;;
    au40) CK=$W/prism_branch40/au/model_final.pt;; ausvp40) CK=$W/prism_branch40/ausvp/model_final.pt;;
    vl4) CK=$W/prism_ukb40/frozen_fn_vl/model_final.pt;; g0) CK=$W/prism_vshape/vshape_g0/model_final.pt;;
    *) echo "모르는 이름 $m"; continue;;
  esac
  echo "=== $m $(date +%H:%M:%S) $CK"
  if [ -f $O/$m/branch_attn.json ] || [ -f $O/$m/dec_attn.json ]; then echo "이미 있음"; continue; fi
  case $m in
    vl4|g0) $PY $PR/dec_attn.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $O/$m | grep -vE "Warning|warn";;
    *)      $PY $PR/branch_attn.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --ckpt $CK --out $O/$m | grep -vE "Warning|warn";;
  esac
done
echo "ALL_DONE $(date +%H:%M:%S)"
