#!/bin/bash
# V-10 (11-28, 5090 GPU 0): Appearance Head(기존 Ours 고정, head만 학습) → 그 예측 PC1로 외형 단계 조향(V-8 ㉥ 쌍1, α = 0.5 × clip(ŝ, ±2)) → 진단 렌더 · 반응 · 질감 → Blender 8지표
#  피험자마다 차례로: SUBS_V10="s01 s06". s01 정답 상한 = V-8 zs_gt(생성 있음, Blender만), s06 정답 상한 = zgt_s06(이번에 생성)
# 사용 (절대 경로로): nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_v10.sh > $W/logs/v10.log 2>&1 < /dev/null &
GPU=0 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
GPU=0; export CUDA_VISIBLE_DEVICES=0; PY=$W/venv/bin/python; O=$W/prism_v8; H=$W/prism_v10; mkdir -p $H/react
F=/home/oem/jhj/fair_eval; V=$F/fair_eval; export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home
evalset() {  # $1 세트 이름, $2 yawg 폴더
  local E=yaw_v10_$1; wait_gpu_free eval_$1
  [ -f $V/out/$E/values.txt ] || bash $V/run_eval.sh $E $2 glb 0 > $F/logs/eval_$E.log 2>&1
  echo "CV|$1 | $(grep -E '^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(date +%H:%M:%S)"
}
react() {  # 세트 목록 (쉼표)
  $PY -I $PR/v8_react_feat.py --diag $O/diag --sets $1 --out $H/react | grep REACT
  $PY -I $PR/v10_render_tex.py --diag $O/diag --sets $1 --out $H/react | grep TEX
}
for SN in ${SUBS_V10:-s01 s06}; do
  SUB=sub-000${SN#s0}; P0=$W/prism_full/eval_ausvp/text/$SN/prompt.json
  echo "=== $SN head 학습 $(date +%H:%M:%S)"; wait_gpu_free v10_head_$SN
  [ -f $H/steer_$SN.json ] || (cd $W/run && $PY $PR/v10_head.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_full/ausvp/model_final.pt --v9 $W/prism_v9/v9_feat.npz --v5 $W/prism_v9/v5_feat.npz \
      --sub $SUB --base_prompt $P0 --out $H)
  [ -f $H/steer_$SN.json ] || { echo "head 실패 $SN"; continue; }
  ST="ah_$SN=$H/steer_$SN.json"
  if [ $SN != s01 ]; then
    $PY $PR/v8_steer_json.py --base $P0 --v5 $W/prism_v9/v5_feat.npz --ann $DR/annotations --mode z --src gt --k 0.5 \
      --plus '{an} with fine surface details and patterns' --minus '{an} with a smooth plain surface' --out $H/zgt_$SN.json
    ST="$ST zgt_$SN=$H/zgt_$SN.json"
    [ -d $O/diag/base_$SN ] || (cd $MG && source $MG/mg_guard.sh >/dev/null && python $PR/v8_diag_render.py --glb $W/prism_yaw/fullausvp_$SN/glb --out $O/diag/base_$SN) 2>&1 | grep -E "DONE|Error"
  fi
  SUBN=$SN STEERS="$ST" bash $PR/scripts/run_v8_steer.sh
  if [ $SN = s01 ]; then react base,zs_gt,zs_pred,ah_s01; else react base_$SN,ah_$SN,zgt_$SN; fi
  echo "DIAG_READY|$SN|$(date +%H:%M:%S)"
  evalset ah_$SN $O/ah_$SN/yawg
  if [ $SN = s01 ]; then evalset zs_gt $O/zs_gt/yawg; else evalset zgt_$SN $O/zgt_$SN/yawg; fi
  echo "SUB_DONE|$SN|$(date +%H:%M:%S)"
done
echo "=== V10_DONE $(date +%H:%M:%S)"
