#!/bin/bash
# P-5 학습(run_p5.sh) 진행 표 — 스텝 / 속도 / 남은 시간 / 끝 시각(KST) + Bank w 엔트로피 · 최댓값. 읽기만 함
# 사용: watch -n 1 "MODE=full bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/status_p5.sh"   (MODE=check 기본)
W=/home/oem/jhj/MinD-3D++; MODE=${MODE:-check}; L=$W/logs/p5_$MODE.log; now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
echo "=== P-5 $MODE · 지금 $(date +%H:%M:%S) · $(grep -a "P-5 $MODE 시작" $L 2>/dev/null)"
grep -aE "^\[bank\]|기다림|다른 사람|Error|Traceback|P5_.*_DONE" $L 2>/dev/null | tail -4 | cut -c1-260
last=$(grep -a "^\[ep " $L 2>/dev/null | tail -1)
if [ -n "$last" ]; then
  s=$(echo "$last" | grep -oE 'step [0-9]+/[0-9]+' | head -1 | tr -dc '0-9/'); cur=${s%/*}; tot=${s#*/}; el=$(echo "$last" | grep -oE ' [0-9]+s ' | tail -1 | tr -dc '0-9')
  if [ -n "$el" ] && [ "$cur" -gt 0 ]; then r=$(( (tot - cur) * el / cur ))
    echo "진행 $cur/$tot · $(awk "BEGIN{printf \"%.2f스텝/초\", $cur/$el}") · 남은 $(( r / 60 ))분 · 끝 $(kst $(( now + r ))) KST"; fi
  echo "$last" | grep -oE 'total [0-9.]+|tok_cos [0-9.]+|proto_acc [0-9.]+|subj_acc [0-9.]+|w_ent [0-9.]+|w_max [0-9.]+|tau [0-9.]+|mem [0-9.]+GB' | tr '\n' ' '; echo
  echo "(균등이면 w_ent = ln 1300 = 7.170, w_max = 1/1300 = 0.00077)"
fi
grep -a "^\[val\]" $L 2>/dev/null | tail -2 | cut -c1-300
echo "--- GPU 1: $(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader) · 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"
