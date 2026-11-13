#!/bin/bash
# 과적합 대책 학습(run_reg.sh) 진행 표 — "진행 / 속도 / 남은 시간(종료 시각, KST)" + 검증 물체([hold]) 요약 (읽기만 함)
#  요약: 지금 = 가장 최근 [hold] 줄, 최저 = 토큰 손실(1 − cos)이 가장 낮았던 스텝. 범주 = 7명 평균 범주 정확도
# 사용: watch -n 1 bash status_reg.sh        (RUNS="base do …"로 바꿀 수 있음)        UKB 인코더판: UKB=1 watch -n 1 bash status_reg.sh
W=/home/oem/jhj/MinD-3D++; UKB=${UKB:-0}; O=$W/prism_reg; LP=prism_reg; DEF="base do dowd aug augdo lin"; [ $UKB = 1 ] && { O=$W/prism_reg_ukb; LP=prism_regukb; DEF="do dowd aug augdo lin"; }
[ "${SET:-}" = branch ] && { O=${OUT:-$W/prism_branch}; LP=${LP:-prism_branch}; DEF="a asv ac ap au"; }   # 11-13 새 구조 비교 (run_branch.sh): SET=branch watch -n 1 bash status_reg.sh
RUNS=${RUNS:-$DEF}; now=$(date +%s); kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
printf "%-7s %-6s %-13s %-12s %-8s %s\n" 이름 단계 진행 속도 남은시간 "종료(KST)"
for n in $RUNS; do
  L=$W/logs/${LP}_$n.log
  [ -f $L ] || { printf "%-7s %-6s\n" $n 대기; continue; }
  read st mx <<< "$(grep -aoE 'step [0-9]+/[0-9]+' $L | tail -1 | tr '/' ' ' | cut -d' ' -f2,3)"
  [ -f $O/$n/model_final.pt ] && { printf "%-7s %-6s %-13s\n" $n 완료 "$mx/$mx"; continue; }
  tail -c 20000 $L | grep -aqE 'Traceback|out of memory' && { printf "%-7s %-6s %s\n" $n 에러 "$(grep -aE 'Error|out of memory' $L | tail -1 | cut -c1-80)"; continue; }
  [ -z "$st" ] && { printf "%-7s %-6s\n" $n 준비; continue; }
  t=$(grep -h "^=== 학습 시작 $n " $W/logs/$LP.log 2>/dev/null | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}$'); e=1
  [ -n "$t" ] && { e=$(( now - $(date -d "$t" +%s) )); [ $e -lt 0 ] && e=$(( e + 86400 )); }; [ $e -lt 1 ] && e=1
  rem=$(( (mx - st) * e / (st > 0 ? st : 1) ))
  printf "%-7s %-6s %-13s %-12s %-8s %s\n" $n 학습 "$st/$mx" "$(awk "BEGIN{printf \"%.1f스텝/초\", $st/$e}")" "$(( rem / 60 ))분" "$(kst $(( now + rem )))"
done
echo "요약 (검증 물체 104개 × 7명, 토큰 손실은 낮을수록 좋음)"
printf "%-7s %-22s %-28s %s\n" 이름 "지금 (스텝 손실 범주)" "최저 손실 (스텝 손실 범주)" "test 범주(기록만)"
for n in $RUNS; do
  C=$O/$n/train_log.csv; [ -f $C ] || continue
  awk -F, -v n=$n 'NR==1{for(i=1;i<=NF;i++){if($i=="tok_cos")c=i; if($i=="proto_acc")a=i}; next}
    $2=="hold"{ls=$1; lc=$c; la=$a; if(bs==""||$c<bc){bs=$1; bc=$c; ba=$a}} $2=="val"{va=$a}
    END{if(ls!="") printf "%-7s %-22s %-28s %s\n", n, sprintf("%d %.4f %.3f", ls, lc, la), sprintf("%d %.4f %.3f", bs, bc, ba), sprintf("%.3f", va)}' $C
done
grep -qE 'REG_DONE|BRANCH_DONE' $W/logs/$LP.log 2>/dev/null && echo "=== 학습 전부 끝남"
echo "지금 $(kst $now) KST | GPU: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ') | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
