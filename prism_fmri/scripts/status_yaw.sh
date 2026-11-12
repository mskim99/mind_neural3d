#!/bin/bash
# 방향 보정 평가(run_yaw_v2.sh) 진행 표 — 세트마다 "진행 / 속도 / 남은 시간(종료 시각, KST)" (읽기만 함, 아무것도 안 고침)
#  단계: 12방위 렌더(물체 수) → 방향 고르기 → 평가 렌더(full만, 물체 수) → 지표 계산 → 완료(values.txt)
#  속도 = 지금 단계에서 실제로 끝난 물체 수 / 그 단계 경과 시간. 아직 안 온 단계는 11-11 실측 평균으로 더함 (추정):
#         12방위 렌더 3.7개/분 · 방향 고르기 4분 · 평가 렌더 2.7개/분 · 지표 계산 12분 (104개 기준, 220개면 비례)
# 사용: watch -n 1 bash status_yaw.sh            (SETS="이름:full|shape:v2|apac …"로 바꿀 수 있음)
W=/home/oem/jhj/MinD-3D++; Y=$W/prism_yaw; L=${L:-$W/logs/yaw_v2.log}   # L = run_yaw_v2.sh 로그 (ablation 7명이면 logs/yaw_abl_subs_g<GPU>.log)
SETS=${SETS:-"vt_s8:full:v2 vt_s2:full:v2 vt_s3:full:v2 vt_s5:full:v2 vt_s6:full:v2 vt_s7:full:v2 ours_AP:shape:v2 ours_APAC220:shape:apac"}
now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
since() { local t=$(grep -h "^=== $1 $2" $L 2>/dev/null | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}$'); [ -z "$t" ] && { echo 0; return; }
          local s=$(( now - $(date -d "$t" +%s) )); [ $s -lt 0 ] && s=$(( s + 86400 )); echo $s; }
npng() { find $1 -name '*.png' 2>/dev/null | wc -l; }
printf "%-14s %-12s %-10s %-10s %-9s %s\n" 세트 단계 진행 속도 남은시간 "종료(KST)"; last=0
for s in $SETS; do
  IFS=: read n mode rt <<< "$s"; R=$W/fair_eval_v2; need=104; [ "$rt" = apac ] && { R=$W/fair_eval_v2_apac; need=220; }
  V=$R/out/yaw_$n/values.txt; k=$(( need * 100 / 104 ))                      # 104개 대비 배율(%)
  tsel=$(( 240 * k / 100 )); tmet=$(( 720 * k / 100 )); tev=0; [ $mode = full ] && tev=$(( need * 60 * 10 / 27 ))
  if [ -s $V ] && grep -q '^FPD' $V; then
    printf "%-14s %-12s %-10s %-10s %-9s %s\n" $n 완료 $need/$need - 0분 "$(kst $(stat -c %Y $V)) 끝남"; continue
  elif [ -d $R/out/yaw_$n ]; then
    e=$(since $n "v2 평가"); d=$(( $(npng $R/out/yaw_$n/render_R2) / 6 ))
    if [ $mode = full ] && [ $d -lt $need ]; then
      st=평가렌더; pr=$d/$need; if [ $d -gt 0 ] && [ $e -gt 0 ]; then sp=$(awk "BEGIN{printf \"%.1f개/분\", $d*60/$e}"); rem=$(( (need - d) * e / d + tmet )); else sp=-; rem=$(( tev + tmet )); fi
    else st=지표계산; pr=-; sp=-; x=$e; [ $mode = full ] && x=$(( now - $(stat -c %Y $R/out/yaw_$n/render_R2 2>/dev/null || echo $now) )); rem=$(( tmet - x )); [ $rem -lt 60 ] && rem=60; fi
  elif [ -f $Y/$n/yawsel.json ]; then st=GLB회전; pr=-; sp=-; rem=$(( tev + tmet ))
  elif [ -d $Y/$n/render ]; then
    e=$(since $n "12방위 렌더"); d=$(( $(npng $Y/$n/render) / 12 ))
    if [ $d -lt $need ]; then st=12방위렌더; pr=$d/$need; if [ $d -gt 0 ] && [ $e -gt 0 ]; then sp=$(awk "BEGIN{printf \"%.1f개/분\", $d*60/$e}"); rem=$(( (need - d) * e / d + tsel + tev + tmet )); else sp=-; rem=$(( need * 600 / 37 + tsel + tev + tmet )); fi
    else st=방향고르기; pr=-; sp=-; rem=$(( tsel + tev + tmet )); fi
  else
    printf "%-14s %-12s %-10s %-10s %-9s %s\n" $n 대기 0/$need - "+$(( (need * 600 / 37 + tsel + tev + tmet) / 60 ))분" "시작 뒤"; continue
  fi
  [ $(( now + rem )) -gt $last ] && last=$(( now + rem ))
  printf "%-14s %-12s %-10s %-10s %-9s %s\n" $n $st $pr $sp "$(( rem / 60 ))분" "$(kst $(( now + rem )))"
done
grep -q YAW_V2_DONE $L 2>/dev/null && echo "=== 전부 끝남" || { [ $last -gt 0 ] && echo "도는 세트 중 가장 늦은 종료: $(kst $last) KST (대기 세트는 시작 뒤 더해짐, 추정)"; }
echo "지금 $(kst $now) KST | GPU: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
