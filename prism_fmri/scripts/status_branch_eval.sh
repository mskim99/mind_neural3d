#!/bin/bash
# 새 구조 평가(run_branch_eval.sh) 진행 표 — GLB 생성 "진행 / 속도 / 남은 시간(종료 시각, KST)" + 방향 보정 · 평가 표(status_yaw.sh) (읽기만 함)
# 사용: watch -n 1 bash status_branch_eval.sh        (NAMES="br_a5k …"로 바꿀 수 있음)
W=/home/oem/jhj/MinD-3D++; O=$W/prism_branch/eval; NAMES=${NAMES:-"br_a5k br_a10k br_au10k br_au20k"}; now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
printf "%-10s %-8s %-9s %-10s %-9s %s\n" 세트 GLB 진행 속도 남은시간 "종료(KST)"; left=0; rate=0
for n in $NAMES; do
  g=$O/$n/glb; d=$(ls $g/*/*.glb 2>/dev/null | wc -l)
  if [ $d -ge 104 ]; then printf "%-10s %-8s %-9s\n" $n 완료 104/104
  elif [ $d -gt 1 ]; then
    t0=$(stat -c %Y $(ls -tr $g/*/*.glb | head -1)); e=$(( now - t0 )); [ $e -lt 1 ] && e=1; rem=$(( (104 - d) * e / (d - 1) )); rate=$(awk "BEGIN{print $e/($d-1)}")
    printf "%-10s %-8s %-9s %-10s %-9s %s\n" $n 생성 $d/104 "$(awk "BEGIN{printf \"%.1f개/분\", ($d-1)*60/$e}")" "$(( rem / 60 ))분" "$(kst $(( now + rem )))"; left=$(( left + rem ))
  else
    if [ "$rate" != 0 ]; then rem=$(awk "BEGIN{printf \"%d\", 104*$rate}"); left=$(( left + rem )); printf "%-10s %-8s %-9s %-10s %-9s %s\n" $n 대기 $d/104 - "+$(( rem / 60 ))분" "$(kst $(( now + left )))"
    else printf "%-10s %-8s %-9s\n" $n 대기 $d/104; fi
  fi
done
echo "--- 방향 보정 · 평가"
L=$W/logs/yaw_branch.log SETS="$(for n in $NAMES; do echo -n "$n:full:v2 "; done)" bash $(dirname $0)/status_yaw.sh
grep -aE "실패|Error|Traceback" $W/logs/branch_eval.log 2>/dev/null | tail -3
grep -q BRANCH_EVAL_DONE $W/logs/branch_eval.log 2>/dev/null && echo "=== 평가 전부 끝남"
