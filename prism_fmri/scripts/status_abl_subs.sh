#!/bin/bash
# Ablation 7명(run_abl_subs.sh) 진행 표 — "진행 / 속도 / 남은 시간(종료 시각, KST)" (읽기만 함)
#  위: GLB 생성 (끝난 세트 수, 세트당 평균 시간 = 로그의 GLB| 줄 실측, 남은 시간) / 아래: 방향 보정 · 평가 (status_yaw.sh, 끝난 세트는 한 줄로 셈)
# 사용: watch -n 1 "ABL='noSVP onlyP noP' bash status_abl_subs.sh"      (5090은 GPU=0 ABL='onlyS noS' 처럼)
W=/home/oem/jhj/MinD-3D++; ABL=${ABL:-"noSVP onlyP noP"}; SS=${SS:-"2 3 5 6 7 8"}; G=${GPU:-1}; A=$W/prism_abl/infer; LG=$W/logs/abl_subs.log
now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
tot=0; done=0; cur=""; SETS=""
for k in $SS; do for n in $ABL; do
  c=$(ls $A/$n/sub-000${k}_keep/glb_text/*/*.glb 2>/dev/null | wc -l); tot=$(( tot + 1 )); SETS="$SETS abl_${n}_s$k:full:v2"
  if [ $c -ge 104 ]; then done=$(( done + 1 )); elif [ $c -gt 0 ]; then cur="$cur ${n}_s$k $c/104"; fi
done; done
avg=$(grep -E '^GLB\|' $LG 2>/dev/null | awk -F'|' '{gsub("s","",$5); if ($5 > 60) {s += $5; n++}} END {if (n) printf "%d", s / n; else print 0}')
if [ $avg -gt 0 ]; then rem=$(( (tot - done) * avg )); echo "GLB 생성: $done/$tot 세트 | 세트당 $(( avg / 60 ))분 (실측) | 남은 $(( rem / 60 ))분 → $(kst $(( now + rem ))) KST | 지금:$cur"
else echo "GLB 생성: $done/$tot 세트 | 속도 아직 없음 (첫 세트 도는 중) | 지금:$cur"; fi
SETS="$SETS" L=$W/logs/yaw_abl_subs_g$G.log bash $(dirname $0)/status_yaw.sh | awk '$2=="완료"{d++; next} $2=="대기"{w++; next} {print} END{print "평가 완료 " d+0 "세트 · 대기 " w+0 "세트"}'
