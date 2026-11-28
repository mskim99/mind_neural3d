#!/bin/bash
# Pro6000 지금 도는 것 한눈에 (11-28): V-10 피험자 6번 + P-5 2만 스텝 — 진행 / 속도 / 남은 시간 / 끝 시각(KST). 읽기만 함
# 사용: watch -n 1 bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/status_now.sh
SD=$(cd $(dirname $0) && pwd); W=/home/oem/jhj/MinD-3D++
echo "################ V-10 피험자 6번 ################"
bash $SD/status_v10_p6.sh 2>/dev/null | grep -v "^--- GPU 1"
echo; echo "################ P-5 Perceptive Bank 2만 스텝 ################"
MODE=full bash $SD/status_p5.sh 2>/dev/null | grep -v "^--- GPU 1"
echo; echo "################ P-5b 학습되는 온도 τ 2만 스텝 ################"
MODE=temp bash $SD/status_p5.sh 2>/dev/null | grep -v "^--- GPU 1\|^(균등"
[ -f $W/logs/p5_eval.log ] && { echo; echo "################ P-5 · P-5b 평가 ################"; bash $SD/status_p5_eval.sh 2>/dev/null | grep -v "^--- GPU 1"; }
echo; echo "--- GPU 1: $(nvidia-smi -i 1 --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader) · 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}') · 부하 $(cut -d' ' -f1-3 /proc/loadavg)"
