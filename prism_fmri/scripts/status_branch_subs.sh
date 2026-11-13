#!/bin/bash
# 새 구조 피험자별 평가(run_branch_subs.sh) 진행 표 — "진행 / 속도 / 남은 시간(종료 시각, KST)" (읽기만 함)
#  위: 피험자별 GLB 생성 / 아래: 방향 보정 · 평가 (status_yaw.sh)
# 사용: watch -n 1 "NAME=aup SS='2 3 5 6' bash status_branch_subs.sh"      (5090은 GPU=0 SS=7 처럼)
W=/home/oem/jhj/MinD-3D++; NAME=${NAME:-aup}; SS=${SS:-"2 3 5 6"}; G=${GPU:-1}; I=$W/prism_branch/subs/$NAME; LG=$W/logs/br_subs_${NAME}_g$G.log
now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
printf "%-12s %-6s %-9s %-10s %-9s %s\n" 세트 GLB 진행 속도 남은시간 "종료(KST)"; SETS=""; nw=0
for k in $SS; do
  n=br_${NAME}_s$k; g=$I/sub-000${k}_keep/glb_text; d=$(ls $g/*/*.glb 2>/dev/null | wc -l); SETS="$SETS $n:full:v2"
  if [ $d -ge 104 ]; then printf "%-12s %-6s %-9s\n" $n 완료 104/104
  elif [ $d -gt 1 ]; then
    t0=$(stat -c %Y $(ls -tr $g/*/*.glb | head -1)); e=$(( now - t0 )); [ $e -lt 1 ] && e=1; rem=$(( (104 - d) * e / (d - 1) ))
    printf "%-12s %-6s %-9s %-10s %-9s %s\n" $n 생성 $d/104 "$(awk "BEGIN{printf \"%.1f개/분\", ($d-1)*60/$e}")" "$(( rem / 60 ))분" "$(kst $(( now + rem )))"
  else printf "%-12s %-6s %-9s\n" $n 대기 $d/104; nw=$(( nw + 1 )); fi
done
avg=$(grep -aE '^GLB\|' $LG 2>/dev/null | awk -F'|' '{gsub("s","",$5); if ($5 > 60) {s += $5; n++}} END {if (n) printf "%d", s / n; else print 0}')
[ $avg -gt 0 ] && [ $nw -gt 0 ] && echo "대기 $nw세트: 세트당 $(( avg / 60 ))분 (실측)씩 더 걸림"
echo "--- 방향 보정 · 평가"
SETS="$SETS" L=$W/logs/yaw_br_subs_${NAME}_g$G.log bash $(dirname $0)/status_yaw.sh
grep -aE "실패|Error|Traceback|없음" $LG 2>/dev/null | tail -3
grep -q BRANCH_SUBS_DONE $LG 2>/dev/null && echo "=== 평가 전부 끝남"
