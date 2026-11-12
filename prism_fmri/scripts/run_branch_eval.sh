#!/bin/bash
# 새 구조(run_branch.sh) 체크포인트 Subject 1 test 104 평가 (11-13, 사용자 승인) — Ours와 같은 방식
#  체크포인트마다: 추론 → 텍스트판 GLB("a {예측 범주}") → 방향 보정(12방위 렌더 → yawsel → 회전) + 평가 (run_yaw_v2.sh 그대로)
#  GLB 생성은 차례로(GPU 약 17 GB), 방향 보정 · 평가는 뒤에서 겹쳐 돎 (run_yaw_v2.sh WAIT=1: GLB 104개가 차면 그 세트 시작)
#  CKS = "이름:체크포인트 …" (기본: a 5천 · 1만, au 1만 · 2만 — 대조는 5090의 공개 인코더 5천 · 4안 1만 · 2만 스텝 값)
#  출력: $W/prism_branch/eval/<이름>/{infer,glb}, $W/prism_yaw/<이름>, 값 $W/fair_eval_v2/out/yaw_<이름>/values.txt (BRANCH_EVAL_DONE)
# 사용: GPU=1 nohup bash run_branch_eval.sh > $W/logs/branch_eval.log 2>&1 < /dev/null & disown
#  진행: watch -n 1 bash status_branch_eval.sh
source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; B=$W/prism_branch; O=$B/eval; TM=microsoft/TRELLIS-text-base; S1=sub-0001
CKS=${CKS:-"br_a5k:$B/a/model_step5000.pt br_a10k:$B/a/model_step10000.pt br_au10k:$B/au/model_step10000.pt br_au20k:$B/au/model_final.pt"}
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cnt() { ls $1/*/*.glb 2>/dev/null | wc -l; }
SETS=""; for c in $CKS; do IFS=: read n ck <<< "$c"; [ -f $ck ] || { echo "체크포인트 없음 $n ($ck)"; exit 1; }; SETS="$SETS $n:$O/$n/glb:$O/$n/infer/${S1}_keep/tokens:full:v2"; done
mkdir -p $O; echo "디스크: $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"
GPU=$CUDA_VISIBLE_DEVICES SETS="$SETS" WAIT=1 PAR=${PAR:-4} bash $PR/scripts/run_yaw_v2.sh > $W/logs/yaw_branch.log 2>&1 &
YP=$!
cd $W/run
for c in $CKS; do
  IFS=: read n ck <<< "$c"; I=$O/$n/infer; g=$O/$n/glb
  echo "=== $n GLB 시작 $(date +%H:%M:%S)"
  [ -f $I/${S1}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $ck --out $I --subs $S1 --perceptive keep 2>&1 | grep -aE "범주|DONE|Error|Traceback"
  [ -f $I/${S1}_keep/components.npz ] || { echo "추론 실패 $n"; continue; }
  $PY $PR/text_prompts.py --comp $I/${S1}_keep/components.npz --out $O/$n/prompt.json | tail -1
  [ $(cnt $g) -ge 104 ] || trel gen --pred $I/${S1}_keep/tokens --slat_text $O/$n/prompt.json --text_model $TM --out $g 2>&1 | grep -aE "GEN_DONE|Error|없음"
  echo "T|$n|GLB $(cnt $g)개|$(date +%H:%M:%S)"
done
wait $YP
grep -a "^YV2|" $W/logs/yaw_branch.log
echo "=== BRANCH_EVAL_DONE $(date +%H:%M:%S)"
