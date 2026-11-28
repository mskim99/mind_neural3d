#!/bin/bash
# 5090 진행 표 (11-28: V-10 마무리 → F-1). 사용: watch -n 1 bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/status_f1.sh
#  줄마다 끝난 수 / 전체 · 속도 · 남은 시간(KST 끝 시각)
W=/home/oem/jhj/MinD-3D++; V=/home/oem/jhj/fair_eval/fair_eval/out; L=$W/logs/f1.log; now=$(date +%s)
eta() {  # $1 끝난 수 $2 전체 $3 시작 epoch $4 단위
  local d=$1 n=$2 t=$((now - $3)); [ $d -le 0 ] || [ $t -le 0 ] && { echo "$d / $n"; return; }
  local r=$(awk "BEGIN{print $d/$t*60}"); local left=$(awk "BEGIN{printf \"%d\", ($n-$d)/($d/$t)}")
  printf "%s / %s · %.1f %s/분 · 남은 %d분 · 끝 %s\n" $d $n $r "$4" $((left/60)) "$(date -d @$((now+left)) +%H:%M)"; }
st() { stat -c %Y "$1" 2>/dev/null || echo $now; }
rend() {  # $1 평가 이름: Blender 렌더 진행
  local R=$V/$1/render_R2; [ -d $R ] || { echo "$1: 시작 전"; return; }
  [ -f $V/$1/values.txt ] && { echo "$1: 끝 — $(grep -E '^R2\|10way|^CD\s' $V/$1/values.txt | tr -s ' \t\n' ' ')"; return; }
  echo "$1: 렌더 $(eta $(ls $R | wc -l) 104 $(st $V/$1/list.txt) 개)"; }
echo "== $(date +%H:%M:%S)  GPU 0: $(nvidia-smi -i 0 --query-gpu=utilization.gpu,memory.used --format=csv,noheader) · 다른 사람 $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | while read p; do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qv /home/oem/jhj && echo $p; done | wc -l)명"
echo "-- V-10 피험자 1번 8지표"; rend yaw_v10_ah_s01; rend yaw_v10_zs_gt
echo "-- F-1 (logs/f1.log)"
[ -f $L ] || { echo "아직 안 뜸"; exit; }
echo "단계: $(grep -a '^===' $L | tail -1)"
for d in $W/prism_f1tune/ausvp $W/prism_f1tune/ausvp_app $W/prism_f1tune/ausvp_app_w01; do [ -f $d/train_log.csv ] && echo "w 고르기 $(basename $d): $(tail -1 $d/train_log.csv | cut -d, -f1) / 1000 스텝"; done
R=$(cat $W/prism_f1/chosen.txt 2>/dev/null)
if [ -n "$R" ]; then
  T=$W/prism_full/$R/train_log.csv; t0=$(date -d "$(grep -a '2만 스텝 7명 학습' $L | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')" +%s 2>/dev/null || echo $now)
  [ -f $T ] && echo "학습 $R: $(eta $(tail -1 $T | cut -d, -f1) 20000 $t0 스텝) · 마지막 hold/val: $(grep -E ',(val|hold),' $T | tail -1 | cut -d, -f1-6)"
  C=$W/prism_full/eval_$R
  for n in 01 02 03 05 06 07 08; do
    g=$(ls $C/text/s$n/glb/*/*.glb 2>/dev/null | wc -l); sg=$(ls $W/prism_f1/st_s$n/glb/*/*.glb 2>/dev/null | wc -l)
    [ $g -eq 0 ] && [ $sg -eq 0 ] && continue
    echo "s$n: 조향 없음 GLB $g/104 · $(rend yaw_f1_s$n | sed 's/^[^:]*: //') | 조향 있음 GLB $sg/104 · $(rend yaw_f1st_s$n | sed 's/^[^:]*: //')"
  done
fi
grep -aE "^CV\||DIAG_READY|S01_DONE|F1_DONE|실패|Traceback" $L | tail -4
# 11-28 18:29: M-1 (run_m1b.sh)
M=$W/prism_2s/ausvp_compsel/train_log.csv; ML=$W/logs/m1b.log
if [ -f $ML ]; then
  echo "-- M-1 (logs/m1b.log) 단계: $(grep -a '^===' $ML | tail -1)"
  t1=$(date -d "$(grep -a 'M-1 학습' $ML | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')" +%s 2>/dev/null || echo $now)
  [ -f $M ] && echo "학습 ausvp_compsel: $(eta $(tail -1 $M | cut -d, -f1) 20000 $t1 스텝)"
  g=$(ls $W/prism_2s/eval_ausvp_compsel/text/s01/glb/*/*.glb 2>/dev/null | wc -l); [ $g -gt 0 ] && echo "M-1 s01: GLB $g/104 · $(rend yaw_2sausvp_compsel_s01 | sed 's/^[^:]*: //')"
  [ -f $W/HOLD_SUBEVAL_f1 ] && echo "F-1 나머지 6명: HOLD (M-1 1번 평가 뒤 자동으로 풀림)"
fi
