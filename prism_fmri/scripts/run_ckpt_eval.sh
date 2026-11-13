#!/bin/bash
# 성능 개선 1단계 — 스텝 수 실험 평가 (11-12, 5090): prism_ckpt/base의 체크포인트마다 Subject 1 test 104
#  추론 → 텍스트판 GLB("a {예측 범주}", Ours와 같은 방식) → 방향 보정(12방위 렌더 → yawsel(자기 토큰) → 회전) → 평가 기준 run_eval.sh
#  체크포인트가 아직 없으면 생길 때까지 기다림. STEPS="5000 final 2000 10000 20000" 순서대로
#  출력: $W/prism_ckpt/infer/<s>/sub-0001_keep, $W/prism_ckpt/text/<s>/glb, $W/prism_yaw/ckpt_<s>, 값 fair_eval/out/yaw_ckpt_<s>/values.txt
# 사용: GPU=0 nohup bash run_ckpt_eval.sh > $W/logs/ckpt_eval.log 2>&1 < /dev/null & disown
#  다른 학습 폴더: CK_ROOT(추론 · GLB 둘 곳) CK_DIR(체크포인트 폴더) TAG(출력 이름 prism_yaw/<TAG>_<s>, yaw_<TAG>_<s>) — 예: UKB 인코더 학습
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; C=${CK_ROOT:-$W/prism_ckpt}; M=${CK_DIR:-$C/base}; TAG=${TAG:-ckpt}; TM=microsoft/TRELLIS-text-base; S1=sub-0001
F=/home/oem/jhj/fair_eval; V=$F/fair_eval; Y=$W/prism_yaw
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home
freewait() { until [ "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 30; done; }   # 11-13: 같은 GPU의 다른 평가(외형 지표 약 20 GB)와 겹쳐 터지지 않게
cd $W/run
for s in ${STEPS:-5000 final}; do
  ck=$M/model_step$s.pt; [ $s = final ] && ck=$M/model_final.pt
  echo "=== $s 대기 $(date +%H:%M:%S)"; until [ -f $ck ]; do sleep 60; done; sleep 30
  I=$C/infer/$s; g=$C/text/$s/glb; O=$Y/${TAG}_$s; E=yaw_${TAG}_$s; mkdir -p $C/text/$s
  echo "=== $s 시작 $(date +%H:%M:%S)"
  [ -f $I/${S1}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $ck --out $I --subs $S1 --perceptive keep | grep -E "범주|DONE|Error"
  $PY $PR/text_prompts.py --comp $I/${S1}_keep/components.npz --out $C/text/$s/prompt.json | tail -1
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || { freewait 21000; trel gen --pred $I/${S1}_keep/tokens --slat_text $C/text/$s/prompt.json --text_model $TM --out $g | grep -E "GEN_DONE|Error|없음"; }
  echo "T|$s|GLB $(ls $g/*/*.glb 2>/dev/null | wc -l)개|$(date +%H:%M:%S)"
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || { echo "!!! $s GLB가 104개가 아님 — 이 스텝은 멈춤 (다시 실행하면 이어서 만듦)"; continue; }
  mkdir -p $O
  if [ ! -f $O/yawsel.json ]; then
    ls $g/*/*.glb > $O/list.txt
    $FE_BLENDER -b -t 4 -P $V/scripts/render.py -- --list $O/list.txt --out_root $O/render --res 512 --light 8000 --camera_dist 1.5 --fov 56 \
      --views $(seq -s, 0 30 330 | sed 's/\([0-9]*\)/\1:30/g') --metallic0 2>&1 | grep -E "FAIL" | head -5
    trel yawsel --frames $O/render --pred $I/${S1}_keep/tokens --out $O --nview 12 | grep -E "YAWSEL_DONE|Error|없음"
  fi
  [ $(ls $O/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/yaw_apply.py --glb $g --sel $O/yawsel.json --out $O/glb | grep YAWAPPLY
  [ -f $V/out/$E/values.txt ] || bash $V/run_eval.sh $E $O/glb glb $GPU > $F/logs/eval_$E.log 2>&1
  echo "CV|$s | $(grep -E '^frame|^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(grep -o '물체별 최소 평균 [0-9.]*' $V/out/$E/logs/frame.log 2>/dev/null) | $(date +%H:%M:%S)"
done
echo "=== CKPT_EVAL_DONE $(date +%H:%M:%S)"
