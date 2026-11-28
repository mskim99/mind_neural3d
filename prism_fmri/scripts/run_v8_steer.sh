#!/bin/bash
# V-8 ㉥ (11-28): 속도 차이 조향으로 외형 생성 → 방향 보정 → 진단 렌더 (학습 없음). 형상 · seed · 기준 문장은 ㉠과 같음
#  STEERS="이름=<α json> …", 선택 NAMES=<물체 목록 txt>(작은 묶음). 생성은 run_v8.sh와 같은 gen.lock으로 줄을 섬
# 사용: STEERS="s_a5=$W/prism_v8/steer/small_a5.json" NAMES=$W/prism_v8/steer/small_names.txt nohup bash run_v8_steer.sh > log 2>&1 &
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
GPU=0; export CUDA_VISIBLE_DEVICES=0; PY=$W/venv/bin/python; TM=microsoft/TRELLIS-text-base; O=${O8:-$W/prism_v8}; SN=${SUBN:-s01}; SEL=${SEL:-$W/prism_yaw/fullausvp_$SN/yawsel.json}   # 11-28 F-1: SEL · TOK · P0 · O8 바꿔 끼울 수 있음   # 11-28 V-10: SUBN=s06 등 다른 피험자 (기본 s01)
TOK=${TOK:-$W/prism_full/eval_ausvp/infer/$SN/sub-000${SN#s0}_keep/tokens}; P0=${P0:-$W/prism_full/eval_ausvp/text/$SN/prompt.json}
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
freewait() { until [ "$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 30; done; }
for kv in ${STEERS:?}; do
  n=${kv%%=*}; j=${kv#*=}; mkdir -p $O/$n
  (
    flock 8; wait_gpu_free v8_$n; freewait 19000
    echo "=== $n 생성 시작 $(date +%H:%M:%S)"
    trel gen --pred $TOK --slat_text $P0 --slat_steer $j --text_model $TM ${NAMES:+--gen_names $NAMES} --out $O/$n/glb | grep -E "GEN_DONE|Error|조향|없음"
  ) 8>$O/gen.lock
  $PY $PR/yaw_apply.py --glb $O/$n/glb --sel $SEL --out $O/$n/yawg | grep -o "YAWAPPLY_DONE [0-9]*개"
  (cd $MG && source $MG/mg_guard.sh >/dev/null && python $PR/v8_diag_render.py --glb $O/$n/yawg --out $O/diag/$n) 2>&1 | grep -E "DONE|Error"
  echo "STEER|$n|GLB $(ls $O/$n/glb | wc -l)|$(date +%H:%M:%S)"
done
echo "=== STEER_DONE $(date +%H:%M:%S)"
