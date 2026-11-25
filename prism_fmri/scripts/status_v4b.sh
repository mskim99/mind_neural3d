#!/bin/bash
# V-4b (run_v4b.sh) 진행 표 — 형상 · GLB · 평가 (진행 / 속도 / 끝나는 시각 KST, 읽기만 함)
# 사용: watch -n 1 bash status_v4b.sh
W=/home/oem/jhj/MinD-3D++; V=$W/prism_v4; L=$W/logs/v4b.log; SD=$(cd $(dirname $0) && pwd); now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
grep -aE "^=== V-4b|멈춤" $L 2>/dev/null | tail -2
echo "--- 0 step 형상 잠재 · GLB (test 104)"
for g in Z0_s01 Z0_s06; do printf "%-8s 잠재 %3s · GLB %3s\n" $g $(ls $V/gen/$g/z 2>/dev/null | wc -l) $(ls $V/gen/$g/glb/*/*.glb 2>/dev/null | wc -l); done
echo "--- 방향 보정 · 평가 (기준 번들)"
SETS="v4b_Z0_s01:full:v2 v4b_Z0_s06:full:v2" FE2=$W/fair_eval_ref L=$W/logs/v4b_yaw.log bash $SD/status_yaw.sh 2>/dev/null | grep -v '^지금'
grep -aE "실패|Error|Traceback|기다림" $L 2>/dev/null | tail -3
grep -q V4B_DONE $L 2>/dev/null && echo "=== V-4b 끝 → $V/v4b_pack.tar"
echo "지금 $(kst $now) KST | GPU: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ') | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
