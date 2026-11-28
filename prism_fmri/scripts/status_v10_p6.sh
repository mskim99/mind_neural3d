#!/bin/bash
# V-10 피험자 6번(run_v10_p6.sh) 진행 표 — 단계마다 진행 / 속도 / 남은 시간 / 끝나는 시각(KST). 읽기만 함
# 사용: watch -n 1 bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/status_v10_p6.sh
W=/home/oem/jhj/MinD-3D++; L=$W/logs/v10_s06.log; H=$W/prism_v10; O=$H/s06; SD=$(cd $(dirname $0) && pwd); now=$(date +%s)
kst() { date -d "@$(( $1 + 32400 ))" +%H:%M; }
cnt() { ls $1/*/*.glb 2>/dev/null | wc -l; }
row() {   # 이름 · 폴더 · 시작 시각(초, 0이면 아직)
  local c=$(cnt $2) t=$3
  if [ $c -ge 104 ]; then printf "%-22s %-8s %s\n" $1 104/104 "끝남 $(kst $(stat -c %Y $(ls -t $2/*/*.glb | head -1)))"
  elif [ $c -gt 0 ] && [ $t -gt 0 ]; then local e=$(( now - t )); local r=$(( (104 - c) * e / c ))
    printf "%-22s %-8s %-10s 남은 %3d분 · 끝 %s\n" $1 $c/104 "$(awk "BEGIN{printf \"%.1f개/분\", $c*60/$e}")" $(( r / 60 )) "$(kst $(( now + r )))"
  else printf "%-22s %-8s %s\n" $1 $c/104 "대기"; fi; }
ts() { local t=$(grep -a "$1" $2 2>/dev/null | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | tail -1); [ -z "$t" ] && { echo 0; return; }
       local s=$(date -d "$t" +%s); [ $s -gt $now ] && s=$(( s - 86400 )); echo $s; }
echo "=== V-10 피험자 6번 (Pro6000 GPU 1) · 지금 $(date +%H:%M:%S) · $(grep -a "V-10 s06 시작" $L 2>/dev/null)"
echo "--- 단계 표시"; grep -aE "^STAGE|^STEER|^DIAG_READY|^CV|^=== V10|실패|없음|Error|Traceback|다른 사람" $L 2>/dev/null | tail -8
echo "--- 3D 생성"
row "기존 Ours" $W/prism_branch/subs/fullausvp/sub-0006_keep/glb_text $(ts "V-10 s06 시작" $L)
row "Head 조향 (ah_s06)" $O/ah_s06/glb $(ts "ah_s06 생성 시작" $L)
row "정답 조향 (zgt_s06)" $O/zgt_s06/glb $(ts "zgt_s06 생성 시작" $L)
echo "--- head 학습: $(grep -aE "^(head|mlp|ridge|고른|V10_HEAD)" $L 2>/dev/null | tail -2 | cut -c1-140 | tr '\n' '|')"
echo "--- 진단 렌더: $(for s in base_s06 ah_s06 zgt_s06; do printf "%s %s/104  " $s $(ls $O/diag/$s 2>/dev/null | wc -l); done)"
echo "--- 기존 Ours 방향 보정 · 8지표"
SETS="br_fullausvp_s6:full:v2" L=$W/logs/yaw_br_subs_fullausvp_g1.log bash $SD/status_yaw.sh 2>/dev/null | tail -4
for n in ah_s06 zgt_s06; do v=$W/fair_eval_ref/out/yaw_v10_$n
  [ -f $v/values.txt ] && echo "8지표 $n 끝" || { [ -d $v ] && echo "8지표 $n 진행 중 (평가 렌더 $(find $v -name '*.png' 2>/dev/null | wc -l)장)"; }; done
echo "--- GPU 1: $(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader) · 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"
