#!/bin/bash
# B안 3단계 (11-10, 5090): TRELLIS 형상 잠재 prior — 잠재 추출(run/ss_all.sh) 끝나면 → prior 학습 → 생성 128 (T 1.0 · 0.8) → 점유 → 검사 · 그림
# 사용: GPU=1 nohup bash run_b_ss.sh > $W/logs/prism_b_ss_prior.log 2>&1 < /dev/null & disown
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_b; S=$O/ss; P=$O/ss_prior${TAG}
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
step() { echo "=== $1 $(date +%H:%M:%S)"; }
cd $W/run
step "0 잠재 추출 기다림"
while pgrep -f "[t]rellis_tools.py ss " >/dev/null; do sleep 30; done
echo "잠재 $(ls $S/z | wc -l)개 · 점유 $(ls $S/occ | wc -l)개"
step "1 실제 검증 물체 기준 (생성 대신 검증 물체 자체)"
mkdir -p $O/ss_valocc; for u in $(shuf -n 64 --random-source=<(yes) $O/prior_u/val_objs.txt); do ln -sf $S/occ/$u.npy $O/ss_valocc/; done
$PY $PR/occ_check.py --train_occ $S/occ --train_objs $O/prior_u/train_objs.txt --val_objs $O/prior_u/val_objs.txt --gen_occ $O/ss_valocc --out $O/ss_valcheck --tag "real val (ceiling)" | grep -E "OCC|범주|Error"
step "2 prior 학습 (${PSTEPS:-20000} 스텝)"
[ -f $P/prior_final.pt ] || $PY $PR/prior_ss.py train --ss $S --objs $O/prior_u/train_objs.txt --out $P --steps ${PSTEPS:-20000} ${PARGS} > $P.log 2>&1
grep -E "\[model\]|\[val\]|Error" $P.log | tail -6
for T in 1.0 0.8; do
  step "3 생성 128 (T $T) → 점유 → 검사"
  G=$P/gen_t$T
  $PY $PR/prior_ss.py sample --ss $S --ckpt $P/prior_final.pt --out $G --n 128 --temp $T | grep -E "생성|Error"
  trel ssdec --pred $G --out $G | grep -E "SS_DONE|Error"
  $PY $PR/occ_check.py --train_occ $S/occ --train_objs $O/prior_u/train_objs.txt --val_objs $O/prior_u/val_objs.txt --gen_occ $G/occ --out $G/check --tag "SS prior${TAG} T$T" | grep -E "OCC|범주|MONTAGE|Error"
done
step "B_SS_DONE"
