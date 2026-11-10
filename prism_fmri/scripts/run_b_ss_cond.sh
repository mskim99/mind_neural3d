#!/bin/bash
# B안 3단계 범주 조건 (11-11, 5090): 형상 잠재 prior를 범주 조건(+ 10% 조건 없음, CFG)으로 학습 → 범주별 생성(gen_k 범주 = k mod 13)
#  → 점유 · 검사(조건 범주 = 가장 가까운 학습 범주 비율) → 앞 32개 glb (외형 = 텍스트판 "a {조건 범주}")
# 사용: GPU=0 TAG=_cL PSTEPS=40000 PARGS="--width 768 --depth 16 --heads 12" nohup bash run_b_ss_cond.sh > $W/logs/prism_b_ss_cond_cL.log 2>&1 < /dev/null & disown
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_b; S=$O/ss; P=$O/ss_prior${TAG:-_c}
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
step() { echo "=== $1 $(date +%H:%M:%S)"; }
cd $W/run
step "1 범주 조건 prior 학습 (${PSTEPS:-40000} 스텝)"
[ -f $P/prior_final.pt ] || $PY $PR/prior_ss.py train --ss $S --objs $O/prior_u/train_objs.txt --out $P --steps ${PSTEPS:-40000} --cond 1 ${PARGS} > $P.log 2>&1
grep -E "\[model\]|Error" $P.log; grep "\[val\]" $P.log | awk 'NR%5==0' | tail -4
for CFG in ${CFGS:-1.0 3.0}; do
  step "2 생성 128 (T 1.0, cfg $CFG) → 점유 → 검사"
  G=$P/gen_cfg$CFG
  $PY $PR/prior_ss.py sample --ss $S --ckpt $P/prior_final.pt --out $G --n 128 --temp 1.0 --cfg $CFG | grep -E "생성|Error"
  trel ssdec --pred $G --out $G | grep -E "SS_DONE|Error"
  $PY $PR/occ_check.py --train_occ $S/occ --train_objs $O/prior_u/train_objs.txt --val_objs $O/prior_u/val_objs.txt --gen_occ $G/occ --gen_cls $G/cls.json --out $G/check --tag "SS cond${TAG} cfg$CFG" | grep -E "OCC|CLS|Error"
  step "3 glb 32 (cfg $CFG)"
  G=$G D=$O/ss_glb${TAG}_cfg$CFG bash $PR/scripts/run_b_ss_glb.sh | grep -E "^z |MONTAGE|Error"
done
step "B_SS_COND_DONE"
