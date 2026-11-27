#!/bin/bash
# V-8 진단 렌더 묶음 (11-28): 세트마다 생성 GLB 104개를 기다림 → 기준 ㉠ yawsel로 방향 보정(<세트>/yawg) → v8_diag_render.py (nvdiffrast, 텍스처 색, 6뷰)
#  base = 기준 ㉠ (prism_yaw/fullausvp_s01/glb, 이미 방향 보정됨)
# 사용: SETS="base gt_detail …" nohup bash run_v8_diag.sh > log 2>&1 &
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
export CUDA_VISIBLE_DEVICES=0; PY=$W/venv/bin/python; O=$W/prism_v8; SEL=$W/prism_yaw/fullausvp_s01/yawsel.json; D=$O/diag; mkdir -p $D
for n in ${SETS:?}; do
  if [ $n = base ]; then g=$W/prism_yaw/fullausvp_s01/glb
  else
    until [ $(ls $O/$n/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ]; do sleep 60; done
    g=$O/$n/yawg; [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/yaw_apply.py --glb $O/$n/glb --sel $SEL --out $g | grep YAWAPPLY
  fi
  (cd $MG && source $MG/mg_guard.sh >/dev/null && python $PR/v8_diag_render.py --glb $g --out $D/$n) 2>&1 | grep -E "DONE|Error|없음"
  echo "DIAG|$n|$(ls $D/$n | wc -l)|$(date +%H:%M:%S)"
done
echo "=== DIAG_ALL_DONE $(date +%H:%M:%S)"
