#!/bin/bash
# 11-28 18:29 총괄(사용자 지시) 순서: M-1을 F-1 나머지 6명보다 먼저 — λ 1e-3 하나만(F-1 1번과 GPU 0을 같이 써서 메모리가 한 개만 들어감)
#  ① M-1 ausvp_compsel 2명(1 · 6) 2만 스텝 → ② 피험자 1번 평가(run_sub_eval NS=01) → ③ F-1 나머지 6명 풀어 줌(HOLD_SUBEVAL_f1 파일을 옮김)
#  ④ F-1 끝(F1_DONE) 기다림 → F-1 AP · APAC → ⑤ M-1 피험자 6번 평가
# 사용 (절대 경로로): nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_m1b.sh > $W/logs/m1b.log 2>&1 < /dev/null &
GPU=0 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
GPU=0; export CUDA_VISIBLE_DEVICES=0; O=$W/prism_2s; N=ausvp_compsel; CK=$O/$N/model_final.pt; C=$O/eval_$N
echo "=== M-1 학습 $N $(date +%H:%M:%S)"
POLITE=0 MAXJ=1 SUBS=sub-0001,sub-0006 OUT=$O LP=prism_2s STEPS=20000 SAVE=10000 GPU=0 RUNS=$N bash $PR/scripts/run_branch.sh
[ -f $CK ] || { echo "M-1 학습 실패"; exit 1; }
echo "=== M-1 피험자 1번 평가 $(date +%H:%M:%S)"
GPU=0 NS="01" CK=$CK CK_ROOT=$C TAG=2s$N bash $PR/scripts/run_sub_eval.sh
echo "M1_S01_DONE $(date +%H:%M:%S)"
[ -f $W/HOLD_SUBEVAL_f1 ] && mv $W/HOLD_SUBEVAL_f1 $W/prism_f1/HOLD_SUBEVAL_f1.released && echo "=== F-1 나머지 6명 풀어 줌 $(date +%H:%M:%S)"
until grep -q F1_DONE $W/logs/f1.log; do sleep 120; done
echo "=== F-1 AP · APAC $(date +%H:%M:%S)"
for X in AP APAC; do GPU=0 X=$X CK=$W/prism_full/ausvp_app/model_final.pt TAG=f1 bash $PR/scripts/run_apac_final.sh; done
echo "=== M-1 피험자 6번 평가 $(date +%H:%M:%S)"
GPU=0 NS="06" CK=$CK CK_ROOT=$C TAG=2s$N bash $PR/scripts/run_sub_eval.sh
echo "=== M1B_DONE $(date +%H:%M:%S)"
