#!/bin/bash
# M-1 (11-28, 5090 GPU 0): 갈래별 채널 gate(Component Selector) — 최종 Ours(ausvp) 2명(1 · 6) 학습 λ 1e-3 · 1e-2 → 검증(hold) 과제 손실로 λ 고름 → 고른 것만 평가
#  V-10 Blender가 끝난 뒤 시작 (logs/v10.log의 V10_DONE), 짧은 시험(60 스텝) → 2만 스텝 두 개를 GPU 0에 같이 → eval_2s_chain.sh
# 사용 (절대 경로로): nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_m1.sh > $W/logs/m1.log 2>&1 < /dev/null &
GPU=0 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
GPU=0; export CUDA_VISIBLE_DEVICES=0; PY=$W/venv/bin/python; O=$W/prism_2s
echo "=== V-10 끝 기다림 $(date +%H:%M:%S)"; until grep -q V10_DONE $W/logs/v10.log; do sleep 60; done
echo "=== 짧은 시험 $(date +%H:%M:%S)"
POLITE=0 TEST=1 SUBS=sub-0001,sub-0006 OUT=$O LP=prism_2s GPU=0 RUNS=ausvp_compsel bash $PR/scripts/run_branch.sh
grep -E "Traceback|Error" $W/logs/prism_2s_test_ausvp_compsel.log && { echo "시험 실패"; exit 1; }
echo "=== 2만 스텝 λ 1e-3 · 1e-2 $(date +%H:%M:%S)"
POLITE=0 MAXJ=2 SUBS=sub-0001,sub-0006 OUT=$O LP=prism_2s STEPS=20000 SAVE=10000 GPU=0 RUNS="ausvp_compsel ausvp_compsel_l2" bash $PR/scripts/run_branch.sh
CH=$($PY - <<'PY'
import csv
best = None
for n, lam in (('ausvp_compsel', 1e-3), ('ausvp_compsel_l2', 1e-2)):
    rows = [r for r in csv.DictReader(open(f'/home/oem/jhj/MinD-3D++/prism_2s/{n}/train_log.csv')) if r['split'] == 'hold']
    if not rows: continue
    r = rows[-1]; task = float(r['total']) - lam * float(r.get('sel', 0) or 0)
    print(f'{n} step {r["step"]} hold 과제 손실 {task:.4f} (total {r["total"]}, sel {r.get("sel")})', file=__import__('sys').stderr)
    if best is None or task < best[1]: best = (n, task)
print(best[0])
PY
)
echo "=== 고른 λ: $CH $(date +%H:%M:%S)"; echo "$CH" > $O/m1_chosen.txt
GPU=0 NAMES=$CH bash $W/eval_2s_chain.sh
echo "=== M1_DONE $(date +%H:%M:%S)"
