#!/bin/bash
# Perceptive 나머지 학습(run_prem.sh) · 물체별 기여 지도(run_attr_obj.sh) 진행 표 — "진행 / 속도 / 남은 시간(종료 시각, KST)" (읽기만 함)
#  학습 속도 = 로그의 마지막 두 "step a/b" 줄이 찍힌 사이(로그 수정 시각 기준이 아니라 train_log.csv 첫 · 끝 줄 수정 간격을 못 쓰므로 로그 시작 시각 대비 평균)
# 사용: watch -n 1 bash status_prem.sh        (RUNS="s05 sv05 …"로 바꿀 수 있음)
W=/home/oem/jhj/MinD-3D++; RUNS=${RUNS:-"s05 s10 sv05 sv10 ctrl ctrl0"}; now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
printf "%-8s %-8s %-14s %-12s %-9s %s\n" 이름 단계 진행 속도 남은시간 "종료(KST)"
for n in $RUNS; do
  L=$W/logs/prism_prem_$n.log; M=$W/prism_prem/$n; I=$W/prism_prem/infer/$n
  [ -f $L ] || { printf "%-8s %-8s\n" $n 대기; continue; }
  if [ -f $M/model_final.pt ]; then
    g=$(ls $I/sub-0001_keep/glb_text/*/*.glb 2>/dev/null | wc -l)
    [ -f $W/prism_prem/judge/$n.json ] && [ $g -ge 104 ] && { printf "%-8s %-8s %-14s\n" $n 완료 "GLB $g/104"; continue; }
    printf "%-8s %-8s %-14s\n" $n "추론·GLB" "GLB $g/104"; continue
  fi
  read st mx <<< "$(grep -oE 'step [0-9]+/[0-9]+' $L | tail -1 | tr '/' ' ' | cut -d' ' -f2,3)"
  [ -z "$st" ] && { printf "%-8s %-8s\n" $n 준비; continue; }
  t0=$(stat -c %W $L 2>/dev/null); [ -z "$t0" ] || [ "$t0" = 0 ] && t0=$(stat -c %Y $M/train_log.csv 2>/dev/null || echo $now)
  e=$(( $(stat -c %Y $L) - t0 )); [ $e -lt 1 ] && e=1
  rem=$(( (mx - st) * e / (st > 0 ? st : 1) ))
  printf "%-8s %-8s %-14s %-12s %-9s %s\n" $n 학습 "$st/$mx" "$(awk "BEGIN{printf \"%.1f스텝/초\", $st/$e}")" "$(( rem / 60 ))분" "$(kst $(( now + rem )))"
done
A=$W/prism_attr/obj_g0; [ -d $A ] && echo "기여 지도: $(ls $A/attr_obj_*.npz 2>/dev/null | wc -l)/7명 $(grep -q ATTR_OBJ_DONE $W/logs/attr_obj.log 2>/dev/null && echo 끝남 || tail -1 $W/logs/attr_obj.log 2>/dev/null | cut -c1-60)"
grep -q PREM_DONE $W/logs/prism_prem.log 2>/dev/null && echo "=== 학습 · GLB 전부 끝남"
echo "지금 $(kst $now) KST | GPU: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
