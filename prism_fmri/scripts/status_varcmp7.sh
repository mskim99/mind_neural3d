#!/bin/bash
# run_var_cmp7.sh 진행 상황 (읽기만 함). 사용: watch -n 1 bash status_varcmp7.sh
W=/home/oem/jhj/MinD-3D++; L=$W/logs/prism_varcmp7.log
SUBS6="sub-0002 sub-0003 sub-0005 sub-0006 sub-0007 sub-0008"
h=$(grep -m1 '^=== 1' $L | awk '{print $NF}'); t0=$(date -d "$h" +%s); [ $t0 -gt $(date +%s) ] && t0=$((t0 - 86400))   # 시작 시각 (자정 넘김 보정)
el=$(( $(date +%s) - t0 )); printf '경과 %d:%02d  (%s)\n' $((el/3600)) $((el%3600/60)) "$(grep -q VARCMP7_DONE $L && echo 완료 || echo 진행 중)"
for m in feat value; do
  I=$W/prism_infer_nop_$m; line="$m |"
  for s in $SUBS6; do
    g=$(ls $I/${s}_keep/glb/*/*.glb 2>/dev/null | wc -l)
    if grep -q "\[$m\] EVAL $s 끝" $L; then st=평가완료
    elif [ -f $W/logs/spp_nop_${m}_$s.log ]; then st="평가중"
    elif [ -d $I/${s}_keep/tokens ]; then st="GLB $g/104"
    else st=대기; fi
    line="$line ${s#sub-00}:$st |"
  done
  echo "$line"
done
echo "--- 최근 로그"; tail -6 $L
for m in feat value; do f=$(ls -t $W/logs/spp_nop_${m}_sub-*.log 2>/dev/null | head -1); [ -n "$f" ] && echo "[$m 평가 $(basename $f .log | sed 's/.*_//')] $(tail -1 $f | cut -c1-120)"; done
