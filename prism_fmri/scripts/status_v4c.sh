#!/bin/bash
# V-4c (run_v4c.sh) 진행 표 — 입력 지문 · GLB · 평가 (진행 / 속도 / 끝나는 시각 KST, 읽기만 함)
# 사용: watch -n 1 bash status_v4c.sh
W=/home/oem/jhj/MinD-3D++; O=$W/prism_v4c; L=$W/logs/v4c.log; SD=$(cd $(dirname $0) && pwd); now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
grep -aE "^=== V-4c|^FP\||멈춤" $L 2>/dev/null | tail -3
echo "--- GLB (test 104)"
for g in orig_s01 orig_s06 match_s01 match_s06; do [ -d $O/$g ] && printf "%-10s 형상 %3s · GLB %3s\n" $g $(ls $O/$g/z 2>/dev/null | wc -l) $(ls $O/$g/glb/*/*.glb 2>/dev/null | wc -l); done
echo "--- 방향 보정 · 평가 (기준 번들)"; ES=""; for g in orig_s01 orig_s06 match_s01 match_s06; do [ -d $O/$g ] && ES="$ES v4c_$g:full:v2"; done
SETS="$ES" FE2=$W/fair_eval_ref L=$W/logs/v4c_yaw.log bash $SD/status_yaw.sh 2>/dev/null | grep -v '^지금'
grep -aE "^ORIG|^MATCH|=== ②|실패|Error|Traceback|기다림" $L 2>/dev/null | tail -4
grep -q V4C_DONE $L 2>/dev/null && echo "=== V-4c 끝 → $O/v4c_pack.tar"
echo "지금 $(kst $now) KST | GPU: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ') | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
