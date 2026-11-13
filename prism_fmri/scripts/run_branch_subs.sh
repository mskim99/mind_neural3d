#!/bin/bash
# 새 구조 체크포인트 하나를 나머지 피험자로 평가 (11-13, 사용자: UKB + new arch. + 2×2 pool 2만 스텝을 전 피험자로)
#  모델은 처음부터 7명 공동 학습 → 재학습 없음. 피험자(SS)마다: 추론 → 문장 "a {예측 범주}" → 텍스트판 GLB 104
#    → 방향 보정 + 평가 8지표 (run_yaw_v2.sh 그대로, WAIT=1로 GLB가 차는 대로 시작) — run_abl_subs.sh와 같은 흐름
#  세트 이름 br_<NAME>_s<피험자 번호> → GLB $W/prism_branch/subs/<NAME>/sub-000N_keep/glb_text, 값 $FE2/out/yaw_br_<NAME>_s<N>/values.txt
#  GLB 생성은 GENPAR 줄로 나눠 같은 GPU에서 동시에 (기본 1, GPU 약 18 GB씩), 평가는 PAR개씩 (기본 4, 렌더가 CPU 코어 약 3.5개씩 씀)
#  실측용 줄: GLB|<NAME>|s<N>|개수|걸린 초|시각
# 사용 (Pro6000): GPU=1 NAME=aup SS="2 3 5 6" GENPAR=2 nohup bash run_branch_subs.sh > $W/logs/br_subs_aup_g1.log 2>&1 < /dev/null & disown
#      (5090):    GPU=0 NAME=aup SS=7 …  /  GPU=1 NAME=aup SS=8 …     (CK를 안 주면 $W/prism_branch/<NAME>/model_final.pt, EVAL=0이면 GLB까지만)
#  진행: watch -n 1 "NAME=aup SS='2 3 5 6' bash status_branch_subs.sh"
source $(dirname $0)/common.sh || exit 1
NAME=${NAME:?NAME=aup 처럼}; CK=${CK:-$W/prism_branch/$NAME/model_final.pt}; SS=${SS:-"2 3 5 6 7 8"}; GENPAR=${GENPAR:-1}; PAR=${PAR:-4}; EVAL=${EVAL:-1}
PY=$W/venv/bin/python; I=$W/prism_branch/subs/$NAME; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}; SD=$(cd $(dirname $0) && pwd); G=$CUDA_VISIBLE_DEVICES
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cnt() { ls $1/*/*.glb 2>/dev/null | wc -l; }
[ -f $CK ] || { echo "체크포인트 없음 $CK"; exit 1; }
mkdir -p $I; cd $W/run
SETS=""; for k in $SS; do d=$I/sub-000${k}_keep; SETS="$SETS br_${NAME}_s$k:$d/glb_text:$d/tokens:full:v2"; done
echo "체크포인트 $CK | 피험자 $SS | GLB $GENPAR줄 · 평가 $PAR개씩 | 디스크: $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"
if [ "$EVAL" = 1 ]; then SETS="$SETS" PAR=$PAR WAIT=1 GPU=$G bash $SD/run_yaw_v2.sh > $W/logs/yaw_br_subs_${NAME}_g$G.log 2>&1 & EV=$!; fi
gen() {   # 피험자 번호
  local k=$1 s=sub-000$1 t0=$(date +%s); local d=$I/${s}_keep
  [ -f $d/components.npz ] && [ $(ls $d/tokens/*.npy 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I --subs $s --perceptive keep 2>&1 | grep -aE "범주|Error|Traceback"
  [ -f $d/components.npz ] || { echo "추론 실패 $s"; return 1; }
  $PY $PR/text_prompts.py --comp $d/components.npz --out $d/prompt_text.json > /dev/null
  [ $(cnt $d/glb_text) -ge 104 ] || trel gen --pred $d/tokens --slat_text $d/prompt_text.json --text_model $TM --out $d/glb_text 2>&1 | grep -aE "GEN_DONE|Error|없음"
  echo "GLB|$NAME|s$k|$(cnt $d/glb_text)개|$(( $(date +%s) - t0 ))s|$(date +%H:%M:%S)"
}
lane() { local i=0; for k in $SS; do [ $(( i % GENPAR )) = $1 ] && gen $k; i=$(( i + 1 )); done; }
J=""; for g in $(seq 0 $(( GENPAR - 1 ))); do lane $g & J="$J $!"; sleep 5; done
wait $J; echo "=== GLB 끝 $(date +%H:%M:%S)"
[ "$EVAL" = 1 ] && { wait $EV; grep -aE "^YV2\|br_${NAME}_s[0-9] \|" $W/logs/yaw_br_subs_${NAME}_g$G.log; }
echo "=== BRANCH_SUBS_DONE $(date +%H:%M:%S)"
