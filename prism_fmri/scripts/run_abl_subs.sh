#!/bin/bash
# Ablation을 7명 기준으로 (11-12, 사용자: 메인 표가 7명 평균이므로 ablation도 7명 평균)
#  ablation 모델은 처음부터 7명 공동 학습 → 재학습 없음. 나머지 피험자(SS)마다: 추론 → 문장 "a {예측 범주}" → 텍스트판 GLB 104
#    → 방향 보정 + 평가 8지표 (run_yaw_v2.sh 그대로, WAIT=1로 GLB가 차는 대로 시작)
#  세트 이름 abl_<이름>_s<피험자 번호> → GLB $W/prism_abl/infer/<이름>/sub-000N_keep/glb_text, 값 $FE2/out/yaw_abl_<이름>_s<N>/values.txt
#  GLB 생성은 GENPAR 줄로 나눠 같은 GPU에서 동시에 (기본 1), 평가는 PAR개씩 (기본 4, 렌더가 CPU 코어 약 3.5개씩 씀)
#  실측용 줄: GLB|<이름>|s<N>|개수|걸린 초|시각  (평가 쪽은 logs/yaw_abl_subs.log의 "=== … 시각" 줄과 YV2| 줄)
# 사용 (Pro6000): GPU=1 ABL="noSVP onlyP noP" nohup bash run_abl_subs.sh > $W/logs/abl_subs.log 2>&1 < /dev/null & disown
#      (5090):    GPU=0 ABL="onlyS noS" …  /  GPU=1 ABL="onlyV noV" …   (EVAL=0이면 GLB까지만)
source $(dirname $0)/common.sh || exit 1
ABL=${ABL:?ABL="noSVP onlyP noP" 처럼}; SS=${SS:-"2 3 5 6 7 8"}; GENPAR=${GENPAR:-1}; PAR=${PAR:-4}; EVAL=${EVAL:-1}
PY=$W/venv/bin/python; A=$W/prism_abl; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}; SD=$(cd $(dirname $0) && pwd)
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cnt() { ls $1/*/*.glb 2>/dev/null | wc -l; }
cd $W/run
SETS=""; LIST=""
for k in $SS; do for n in $ABL; do   # 피험자 순으로 돌려 한 피험자의 ablation이 먼저 다 모이게
  [ -f $A/$n/model_final.pt ] || { echo "체크포인트 없음 $A/$n/model_final.pt"; exit 1; }
  d=$A/infer/$n/sub-000${k}_keep; SETS="$SETS abl_${n}_s$k:$d/glb_text:$d/tokens:full:v2"; LIST="$LIST $n:$k"
done; done
echo "세트 $(echo $LIST | wc -w)개: $LIST"
if [ "$EVAL" = 1 ]; then SETS="$SETS" PAR=$PAR WAIT=1 GPU=$CUDA_VISIBLE_DEVICES bash $SD/run_yaw_v2.sh > $W/logs/yaw_abl_subs_g$CUDA_VISIBLE_DEVICES.log 2>&1 & EV=$!; fi
gen() {   # 이름 · 피험자 번호
  local n=$1 k=$2 I=$A/infer/$1 s=sub-000$2 t0=$(date +%s); local d=$I/${s}_keep
  [ -f $d/components.npz ] && [ $(ls $d/tokens/*.npy 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $A/$n/model_final.pt --out $I --subs $s --perceptive keep | grep -E "범주|Error"
  [ -f $d/components.npz ] || { echo "추론 실패 $n $s"; return 1; }
  $PY $PR/text_prompts.py --comp $d/components.npz --out $d/prompt_text.json > /dev/null
  [ $(cnt $d/glb_text) -ge 104 ] || trel gen --pred $d/tokens --slat_text $d/prompt_text.json --text_model $TM --out $d/glb_text | grep -E "GEN_DONE|Error|없음"
  echo "GLB|$n|s$k|$(cnt $d/glb_text)개|$(( $(date +%s) - t0 ))s|$(date +%H:%M:%S)"
}
lane() { local i=0; for x in $LIST; do [ $(( i % GENPAR )) = $1 ] && gen ${x%%:*} ${x##*:}; i=$(( i + 1 )); done; }
J=""; for g in $(seq 0 $(( GENPAR - 1 ))); do lane $g & J="$J $!"; sleep 5; done
wait $J; echo "=== GLB 끝 $(date +%H:%M:%S)"
[ "$EVAL" = 1 ] && { wait $EV; grep -E "^YV2\|abl_[A-Za-z]+_s[0-9] \|" $W/logs/yaw_abl_subs_g$CUDA_VISIBLE_DEVICES.log; }
echo "=== ABL_SUBS_DONE $(date +%H:%M:%S)"
