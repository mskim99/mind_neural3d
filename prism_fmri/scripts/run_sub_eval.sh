#!/bin/bash
# 체크포인트 하나로 여러 피험자 test 104 평가 (11-12, 5090) — 7명 평균용
#  피험자마다: 추론 → 텍스트판 GLB("a {예측 범주}") → [뒤에서 이어서] 방향 보정(12방위 렌더 → yawsel → 회전) → 평가 기준 run_eval.sh
#  GLB 생성만 GPU를 크게 쓰므로(약 17 GB) 피험자를 차례로 만들고, 렌더 · 평가는 뒤에서 겹쳐 돌림
#  필수: CK(체크포인트 파일) CK_ROOT(추론 · GLB 둘 곳) TAG(출력 이름)   NS="02 03 05" (피험자 번호)
#  HOLD_DIR: 주면 그 폴더(…/glb)에 GLB 104개가 찰 때까지 기다린 뒤 시작 (같은 GPU의 다른 GLB 생성과 안 겹치게)
#  출력: $CK_ROOT/infer/s<번호>/sub-00<번호>_keep, $CK_ROOT/text/s<번호>/glb, $W/prism_yaw/<TAG>_s<번호>, 값 fair_eval/out/yaw_<TAG>_s<번호>/values.txt
# 사용: GPU=1 NS="06 07 08" CK=… CK_ROOT=… TAG=ukbvl nohup bash run_sub_eval.sh > $W/logs/sub_eval_g1.log 2>&1 < /dev/null & disown
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; C=${CK_ROOT:?}; ck=${CK:?}; TAG=${TAG:?}; TM=microsoft/TRELLIS-text-base
F=/home/oem/jhj/fair_eval; V=$F/fair_eval; Y=$W/prism_yaw
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home
freewait() { until [ "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 30; done; }
post() {   # $1 = 피험자 번호: 방향 보정 → 평가
  local n=$1 S=sub-00$1; local I=$C/infer/s$n g=$C/text/s$n/glb O=$Y/${TAG}_s$n E=yaw_${TAG}_s$n
  mkdir -p $O
  if [ ! -f $O/yawsel.json ]; then
    ls $g/*/*.glb > $O/list.txt
    $FE_BLENDER -b -t 4 -P $V/scripts/render.py -- --list $O/list.txt --out_root $O/render --res 512 --light 8000 --camera_dist 1.5 --fov 56 \
      --views $(seq -s, 0 30 330 | sed 's/\([0-9]*\)/\1:30/g') --metallic0 2>&1 | grep -E "FAIL" | head -5
    freewait 6000; trel yawsel --frames $O/render --pred $I/${S}_keep/tokens --out $O --nview 12 | grep -E "YAWSEL_DONE|Error|없음"
  fi
  [ $(ls $O/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/yaw_apply.py --glb $g --sel $O/yawsel.json --out $O/glb | grep YAWAPPLY
  [ -f $V/out/$E/values.txt ] || bash $V/run_eval.sh $E $O/glb glb $GPU > $F/logs/eval_$E.log 2>&1
  echo "CV|s$n | $(grep -E '^frame|^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(grep -o '물체별 최소 평균 [0-9.]*' $V/out/$E/logs/frame.log 2>/dev/null) | $(date +%H:%M:%S)"
}
cd $W/run
[ -n "$HOLD_DIR" ] && { echo "=== 대기: $HOLD_DIR 104개 $(date +%H:%M:%S)"; until [ $(ls $HOLD_DIR/*/*.glb 2>/dev/null | wc -l) -ge 104 ]; do sleep 30; done; sleep 60; }
for n in ${NS:?}; do
  S=sub-00$n; I=$C/infer/s$n; g=$C/text/s$n/glb; mkdir -p $C/text/s$n
  echo "=== s$n 시작 $(date +%H:%M:%S)"
  [ -f $I/${S}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $ck --out $I --subs $S --perceptive keep | grep -E "범주|DONE|Error"
  $PY $PR/text_prompts.py --comp $I/${S}_keep/components.npz --out $C/text/s$n/prompt.json | tail -1
  freewait 19000
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/${S}_keep/tokens --slat_text $C/text/s$n/prompt.json --text_model $TM --out $g | grep -E "GEN_DONE|Error|없음"
  echo "T|s$n|GLB $(ls $g/*/*.glb 2>/dev/null | wc -l)개|$(date +%H:%M:%S)"
  post $n &
done
wait
echo "=== SUB_EVAL_DONE $(date +%H:%M:%S)"
