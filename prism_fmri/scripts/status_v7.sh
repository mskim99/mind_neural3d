#!/bin/bash
# V-7 평가(run_v7_eval.sh) 진행 표 — 단계마다 진행 / 속도 / 남은 시간 / 끝나는 시각(KST). 읽기만 함
# 사용: watch -n 1 bash status_v7.sh
W=/home/oem/jhj/MinD-3D++; L=$W/logs/v7_eval.log; I=$W/prism_branch/subs; SD=$(cd $(dirname $0) && pwd); now=$(date +%s)
kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }; ago() { local s=$(( now - $(date -d "$1" +%s) )); [ $s -lt 0 ] && s=$(( s + 86400 )); echo $s; }
N="ausvp_vtgt_bandenergy_s1 ausvp_s1"; last=0
t0=$(grep -a "학습 끝 확인" $L 2>/dev/null | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}$')
[ -z "$t0" ] && { echo "평가 시작 전 (학습 끝 기다림) — $(grep -a "V-7 평가 시작" $L 2>/dev/null)"; exit 0; }
e=$(ago $t0); echo "평가 시작 $(kst $(( now - e ))) KST · 지난 $(( e / 60 ))분"
echo "--- ① 3D 생성 (추론 포함)"
printf "%-26s %-9s %-10s %-9s %s\n" 모델 진행 속도 남은시간 "끝(KST)"
for n in $N; do c=$(ls $I/$n/sub-0001_keep/glb_text/*/*.glb 2>/dev/null | wc -l)
  if [ $c -ge 104 ]; then printf "%-26s %-9s %-10s %-9s %s\n" $n 104/104 - 0분 "$(kst $(stat -c %Y $(ls -t $I/$n/sub-0001_keep/glb_text/*/*.glb | head -1))) 끝남"
  elif [ $c -gt 0 ]; then r=$(( (104 - c) * e / c )); [ $(( now + r )) -gt $last ] && last=$(( now + r ))
    printf "%-26s %-9s %-10s %-9s %s\n" $n $c/104 "$(awk "BEGIN{printf \"%.1f개/분\", $c*60/$e}")" "$(( r / 60 ))분" "$(kst $(( now + r )))"
  else printf "%-26s %-9s %-10s %-9s %s\n" $n 0/104 - - 추론중; fi; done
echo "--- ② 방향 보정 · 8지표 (생성되는 대로 따라감)"
for n in $N; do SETS="br_${n}_s1:full:v2" FE2=$W/fair_eval_ref L=$W/logs/yaw_br_subs_${n}_g1.log bash $SD/status_yaw.sh 2>/dev/null | grep -vE '^지금|^세트|가장 늦은'; done
echo "--- ③ 범주 안 고르기 · 렌더 특징 (② 끝난 뒤, 약 5 ~ 10분 추정)"
if grep -qa "8지표 끝" $L; then grep -aE "^\[[0-9]+/|IDENT\||V6_IDENT_DONE|Error" $W/logs/v7_ident.log 2>/dev/null | tail -3 | cut -c1-140; else echo "대기"; fi
grep -aE "^PACK|V7_DONE|멈춤|실패" $L | tail -2
grep -qa V7_DONE $L || { [ $last -gt 0 ] && echo "생성 끝 추정 $(kst $last) KST → 전체 끝(묶음) 추정 $(kst $(( last + 3900 ))) KST (생성 뒤 방향 고르기 4분 · 평가 렌더 104개 약 38분 · 지표 12분 · 범주 안 고르기 10분, 11-11 실측 속도)"; }
echo "지금 $(kst $now) KST | GPU: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')| 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
