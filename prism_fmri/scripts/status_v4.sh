#!/bin/bash
# V-4 1단계 (run_v4.sh) 진행 표 — 준비 · 학습(step / 속도 / 끝나는 시각 KST) · 형상 · GLB · 평가 (읽기만 함)
# 사용: watch -n 1 bash status_v4.sh
W=/home/oem/jhj/MinD-3D++; V=$W/prism_v4; L=$W/logs/v4.log; SD=$(cd $(dirname $0) && pwd); now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
grep -aE "^=== V-4|^코드|멈춤" $L 2>/dev/null | tail -2
echo "--- 준비"; grep -aE "^INFER\||^PREP\|" $L 2>/dev/null | tail -5
echo "--- 학습 (cross-attention만, A = 토큰만 · B = 토큰 + V)"
for n in A B; do f=$W/logs/v4_train_$n.log; l=$(grep -aE '^\[train' $f 2>/dev/null | tail -1)
  if grep -q TRAIN_DONE $f 2>/dev/null; then echo "$n 완료 | ${l#*] }"; elif [ -n "$l" ]; then echo "$n 진행 | ${l#*] }"; elif [ -f $f ]; then echo "$n 시작 중 | $(grep -aE '^\[train\]|Error' $f | tail -1 | cut -c1-150)"; else echo "$n 대기"; fi; done
echo "--- 형상 잠재 · GLB (test 104, 상한 104)"
for g in A_s01 A_s06 B_s01 B_s06 oracle; do printf "%-8s 잠재 %3s · GLB %3s\n" $g $(ls $V/gen/$g/z 2>/dev/null | wc -l) $(ls $V/gen/$g/glb/*/*.glb 2>/dev/null | wc -l); done
echo "--- 방향 보정 · 평가 (기준 번들)"; ES=""; for g in A_s01 A_s06 B_s01 B_s06 oracle; do ES="$ES v4_$g:full:v2"; done
SETS="$ES" FE2=$W/fair_eval_ref L=$W/logs/v4_yaw.log bash $SD/status_yaw.sh 2>/dev/null | grep -v '^지금'
grep -aE "실패|Error|Traceback|기다림" $L 2>/dev/null | tail -3
grep -q V4_DONE $L 2>/dev/null && echo "=== V-4 1단계 끝 → $V/v4_pack.tar"
echo "지금 $(kst $now) KST | GPU: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ') | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
