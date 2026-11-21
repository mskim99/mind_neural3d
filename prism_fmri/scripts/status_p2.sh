#!/bin/bash
# P-2 (run_p2.sh) 진행 표 — 학습 · 추론 · GLB · 방향 보정 평가 각각 "진행 / 속도 / 남은 시간(종료 시각, KST)" (읽기만 함)
# 사용: watch -n 1 bash status_p2.sh
W=/home/oem/jhj/MinD-3D++; P=$W/prism_p2; L=$W/logs/p2.log; SD=$(cd $(dirname $0) && pwd)
M1=ausvp_sharedsv; M2=ausvp_sharedsv_regionw; M3=ausvp_8sub; now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
SETS0="$M1:s04_nopair $M2:s04_nopair $M1:s04_pair1300as01 $M2:s04_pair1300as01 $M2:s04_pair1300as02 $M3:s04 $M1:AP_nopair $M2:AP_nopair $M1:APAC220_nopair $M2:APAC220_nopair $M1:s01 $M1:s06 $M2:s01 $M2:s06"
ALL="$SETS0 ${ADD:-}"
setdir() { local m=$1 c=$2   # run_p2.sh와 같음
  case $c in s0[0-9]) echo $P/infer/$m/sub-00${c#s}_keep;; s04_nopair) echo $P/infer/$m/sub-0004_new_keep;; AP_nopair) echo $P/infer/$m/sub-0009_new_keep;;
    APAC220_nopair) echo $P/infer/$m/sub-0011_new_keep;; s04_pair*as*) local n=${c#s04_pair}; echo $P/align/$m/as${n#*as}_n${n%as*}/n${n%as*}/sub-0004_as${n#*as}_keep;; esac; }
grep -aE "^=== P-2 시작|기준 번들|다른 사람 작업" $L 2>/dev/null | tail -3
echo "--- 학습"; SET=branch OUT=$P LP=prism_p2 RUNS="$M1 $M2 $M3" bash $SD/status_reg.sh 2>/dev/null | sed -n '1,4p'
echo "--- 추론 (범주 정확도) · GLB"
for s in $ALL; do n=p2_${s%%:*}_${s#*:}; k=104; case $s in *APAC220*) k=220;; esac
  c=$(grep -a "^CAT|$n|" $L 2>/dev/null | tail -1 | cut -d'|' -f3 | grep -oE '범주 정확도 [0-9.na]+|test 범주 [0-9.]+' | tail -1)
  d=$(setdir ${s%%:*} ${s#*:}); g=$(ls $d/glb_text/*/*.glb 2>/dev/null | wc -l)
  printf "%-44s %-18s GLB %s\n" $n "${c:-추론 대기}" "$g/$k$([ $g -ge $k ] && echo ' 완료')"
done
ES=""; for s in $ALL; do c=${s#*:}; m=full:v2; case $c in APAC220*) m=shape:apac;; esac; ES="$ES p2_${s%%:*}_$c:$m"; done
echo "--- 방향 보정 · 평가 (기준 번들)"; SETS="$ES" FE2=$W/fair_eval_ref FEA=$W/fair_eval_ref_apac L=$W/logs/p2_yaw.log bash $SD/status_yaw.sh 2>/dev/null | grep -v '^지금'
grep -aE "실패|Error|Traceback|없음|기다림" $L 2>/dev/null | tail -3
grep -q P2_DONE $L 2>/dev/null && echo "=== P-2 전부 끝남 → $P/p2_pack.tar"
echo "지금 $(kst $now) KST | GPU: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ') | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
