#!/bin/bash
# 추론 폴더(tokens/ + components.npz) 여러 개를 차례로 3D 평가 (11-20, 5090) — 새 피험자 정렬(align_infer.py · train_adapt.py --save) 결과용
#  폴더마다: 텍스트판 GLB("a {예측 범주}") → [뒤에서 이어서] 방향 보정(12방위 렌더 → yawsel → 회전) → 평가 기준 run_eval.sh  (run_sub_eval.sh와 같은 명령)
#  필수: SETS="<이름>=<추론 폴더> …"  (이름 = 출력 꼬리표, 값은 fair_eval/out/yaw_<이름>/values.txt)
# 사용: GPU=0 SETS="al_n1300_02as01=$W/prism_align/x/n1300/sub-0002_as01_keep" nohup bash run_dir_eval.sh > $W/logs/dir_eval.log 2>&1 < /dev/null & disown
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; TM=microsoft/TRELLIS-text-base; F=/home/oem/jhj/fair_eval; V=$F/fair_eval; Y=$W/prism_yaw
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home
freewait() { until [ "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 30; done; }
post() {   # $1 = 이름, $2 = 추론 폴더
  local d=$2 g=$2/glb O=$Y/$1 E=yaw_$1
  mkdir -p $O
  if [ ! -f $O/yawsel.json ]; then
    ls $g/*/*.glb > $O/list.txt
    $FE_BLENDER -b -t 4 -P $V/scripts/render.py -- --list $O/list.txt --out_root $O/render --res 512 --light 8000 --camera_dist 1.5 --fov 56 \
      --views $(seq -s, 0 30 330 | sed 's/\([0-9]*\)/\1:30/g') --metallic0 2>&1 | grep -E "FAIL" | head -5
    freewait 6000; trel yawsel --frames $O/render --pred $d/tokens --out $O --nview 12 | grep -E "YAWSEL_DONE|Error|없음"
  fi
  [ $(ls $O/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/yaw_apply.py --glb $g --sel $O/yawsel.json --out $O/glb | grep YAWAPPLY
  [ -f $V/out/$E/values.txt ] || bash $V/run_eval.sh $E $O/glb glb $GPU > $F/logs/eval_$E.log 2>&1
  echo "CV|$1 | $(grep -E '^frame|^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(date +%H:%M:%S)"
}
cd $W/run
for kv in ${SETS:?}; do
  n=${kv%%=*}; d=${kv#*=}; [ -f $d/components.npz ] || { echo "없음 $d"; continue; }
  echo "=== $n 시작 $(date +%H:%M:%S)"
  $PY $PR/text_prompts.py --comp $d/components.npz --out $d/prompt.json | tail -1
  freewait 19000
  [ $(ls $d/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $d/tokens --slat_text $d/prompt.json --text_model $TM --out $d/glb | grep -E "GEN_DONE|Error|없음"
  echo "T|$n|GLB $(ls $d/glb/*/*.glb 2>/dev/null | wc -l)개|$(date +%H:%M:%S)"
  post $n $d &
done
wait
echo "=== DIR_EVAL_DONE $(date +%H:%M:%S)"
