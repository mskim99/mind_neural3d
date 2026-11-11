#!/bin/bash
# B안 → PRISM 붙이기 평가 (11-11, 5090): Subject 1 test 104, fMRI 예측 범주(prism_img10k sub-0001 Semantic top-1)
#  → 범주 조건 형상 잠재 prior(_cL, CFG 3)로 물체마다 형상 잠재 하나 → TRELLIS 2단계 텍스트판 "a {예측 범주}" → glb → 공정평가 잠정 R2 · P2 · FPD
#  대조: A 텍스트판(같은 예측 범주 · 같은 문장, 형상만 PRISM 토큰) = fe104/glb_text, 이미 잼
# 사용: GPU=0 nohup bash run_b_ss_fe.sh > $W/logs/prism_b_ss_fe.log 2>&1 < /dev/null & disown
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_b; INF=$W/prism_img10k/infer/sub-0001_keep; F=$W/prism_img10k/fe104; D=$O/ss_fe${TAG:-_cL}
CK=${CK:-$O/ss_prior_cL/prior_final.pt}; FE=/home/oem/jhj/fair_eval
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
step() { echo "=== $1 $(date +%H:%M:%S)"; }
mkdir -p $D; cd $W/run
step "0 예측 범주"
$PY -c "
import numpy as np, json; z = np.load('$INF/components.npz'); C = [str(c) for c in z['classes']]
J = {str(k).split('/')[-1]: C[int(c)] for k, c in zip(z['name'], z['cls_pred'])}
json.dump(J, open('$D/pred_cls.json', 'w'), indent=1); print('물체', len(J), '| 범주 정확도', round(float((z['cls'] == z['cls_pred']).mean()), 3))"
step "1 형상 잠재 (CFG ${CFG:-3.0})"
$PY $PR/prior_ss.py sample --ss $O/ss --ckpt $CK --out $D/z --cls_json $D/pred_cls.json --temp 1.0 --cfg ${CFG:-3.0} | grep -E "생성|Error"
step "2 glb (외형 = 텍스트판, A 텍스트판과 같은 문장)"
trel gen --ss_z $D/z --slat_text $F/prompt.json --out $D/glb | grep -E "GEN_DONE|Error|없음|빈 형상"
for k in 1 2 3; do   # 빈 형상은 그 물체만 seed를 바꿔 다시 뽑음 (정답을 보지 않는 규칙)
  $PY -c "
import json, os; J = json.load(open('$D/pred_cls.json')); M = {n: c for n, c in J.items() if not os.path.exists(f'$D/glb/{n}/{n}.glb')}
json.dump(M, open('$D/retry.json', 'w')); print('빈 형상 다시 뽑기', len(M))"
  [ "$(cat $D/retry.json)" = "{}" ] && break
  $PY $PR/prior_ss.py sample --ss $O/ss --ckpt $CK --out $D/z --cls_json $D/retry.json --temp 1.0 --cfg ${CFG:-3.0} --seed $k | grep -E "생성|Error"
  trel gen --ss_z $D/z --slat_text $F/prompt.json --out $D/glb | grep -E "GEN_DONE|Error|빈 형상"
done
echo "glb $(ls $D/glb/*/*.glb | wc -l)"
step "3 공정평가 (잠정 R2 · P2 · FPD)"
bash $FE/run_check.sh B_ss${TAG:-_cL}_s1 $D/glb glb $GPU > $FE/logs/B_ss${TAG:-_cL}_s1.log 2>&1
tail -9 $FE/logs/B_ss${TAG:-_cL}_s1.log
step "B_SS_FE_DONE"
