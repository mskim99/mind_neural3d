#!/bin/bash
# V-8 평가 다시 (11-28): run_v8.sh 첫 판이 run_eval에 빈 GPU 번호를 넘겨 렌더가 CPU로 · 텍스처 지표 실패 → 새 이름 yaw_v8g_<세트>로 GPU 0에서 다시 잼
#  세트마다: 생성 GLB 104개를 기다림 → 기준 ㉠ yawsel로 방향 보정(<세트>/yawg) → run_eval.sh (GPU 0)
# 사용: LANE="gt_detail gt_blur …" nohup bash run_v8_eval.sh > log 2>&1 &
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
GPU=0; export CUDA_VISIBLE_DEVICES=0
PY=$W/venv/bin/python; F=/home/oem/jhj/fair_eval; V=$F/fair_eval; O=$W/prism_v8; SEL=$W/prism_yaw/fullausvp_s01/yawsel.json
export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home
for n in ${LANE:?}; do
  until [ $(ls $O/$n/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ]; do sleep 60; done
  E=yaw_v8g_$n; echo "=== $n 평가 시작 $(date +%H:%M:%S)"
  [ $(ls $O/$n/yawg/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/yaw_apply.py --glb $O/$n/glb --sel $SEL --out $O/$n/yawg | grep YAWAPPLY
  [ -f $V/out/$E/values.txt ] || bash $V/run_eval.sh $E $O/$n/yawg glb 0 > $F/logs/eval_$E.log 2>&1
  echo "CV|$n | $(grep -E '^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(date +%H:%M:%S)"
done
echo "=== LANE_DONE $(date +%H:%M:%S)"
