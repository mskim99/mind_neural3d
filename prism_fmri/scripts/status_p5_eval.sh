#!/bin/bash
# P-5 평가(run_p5_eval.sh) 진행 표 — 모델마다 GLB(끝난 수/전체 · 속도 · 끝 시각 KST) · 평가 세트(끝난 수/9) · 평가 렌더 장수. 읽기만 함
# 사용: watch -n 1 bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/status_p5_eval.sh
W=/home/oem/jhj/MinD-3D++; L=$W/logs/p5_eval.log; now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
cnt() { ls $1/*/*.glb 2>/dev/null | wc -l; }
ts() { local t=$(grep -a "$1" $L 2>/dev/null | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1); [ -z "$t" ] && { echo 0; return; }; local s=$(date -d "$t" +%s); [ $s -gt $now ] && s=$(( s - 86400 )); echo $s; }
echo "=== P-5 평가 · 지금 $(kst $now) KST · $(grep -a "P-5 평가 시작" $L 2>/dev/null | cut -c1-120)"
grep -aE "주의|Error|Traceback|실패|다른 사람" $L $W/logs/p5ev_*.log 2>/dev/null | tail -3 | cut -c1-200
printf "%-10s %-16s %-10s %-14s %-10s %s\n" 모델 "GLB 7명" 속도 "끝(KST)" "AP·APAC" "평가 끝/9 · 렌더"
for n in p5bank p5temp fullausvp; do
  grep -q "$n" $L 2>/dev/null || continue
  t=$(ts "=== $n 시작"); g=0; for k in 1 2 3 5 6 7 8; do g=$(( g + $(cnt $W/prism_branch/subs/$n/sub-000${k}_keep/glb_text) )); done
  a1=$(cnt $W/prism_p5/apac_$n/sub-0009_as01_keep/glb); a2=$(cnt $W/prism_p5/apac_$n/sub-0011_as01_keep/glb)
  if [ $t -eq 0 ]; then sp="대기"; eta="모델 기다림"
  elif [ $g -ge 728 ]; then sp="-"; eta="끝남"
  elif [ $g -gt 0 ]; then e=$(( now - t )); r=$(( (728 - g) * e / g )); sp=$(awk "BEGIN{printf \"%.1f개/분\", $g*60/$e}"); eta="$(kst $(( now + r ))) ($(( r / 60 ))분)"
  else sp="-"; eta="추론 중"; fi
  v=0; for k in 1 2 3 5 6 7 8; do [ -f $W/fair_eval_ref/out/yaw_br_${n}_s$k/values.txt ] && v=$(( v + 1 )); done
  [ -f $W/fair_eval_ref/out/yaw_${n}_AP/values.txt ] && v=$(( v + 1 )); [ -f $W/fair_eval_ref_apac/out/yaw_${n}_APAC220/values.txt ] && v=$(( v + 1 ))
  rp=$(find $W/fair_eval_ref/out -path "*yaw_br_${n}_s*" -name '*.png' 2>/dev/null | wc -l)
  printf "%-10s %-16s %-10s %-14s %-10s %s\n" $n "$g/728" "$sp" "$eta" "$a1/104 $a2/220" "$v/9 · 렌더 ${rp}장"
done
echo "(평가 한 세트 = 12방위 렌더 약 28분 + 방향 고르기 4분 + 평가 렌더 약 38분 + 지표 12분, 모델마다 3세트씩 동시 — 11-11 실측 평균, 추정)"
grep -aE "^GLB7|^GLB\||^MODEL_DONE|P5_EVAL_DONE" $L 2>/dev/null | tail -5
echo "--- GPU 1: $(nvidia-smi -i 1 --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader) · 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"
