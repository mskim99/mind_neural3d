#!/bin/bash
# run_fig.sh 진행 · 남은 시간 (11-06). 사용: watch -n 1 bash $PR/scripts/status_fig.sh
#  학습 남은 시간 = 지금까지 평균 속도(경과 초 / 스텝)로 계산. 학습 뒤 GLB · 평가는 약 1시간으로 잡음 (추정)
L=/home/oem/jhj/MinD-3D++/logs; AFTER=3600
fmt() { printf '%d시간 %02d분 %02d초' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
grep -E "===|학습|GLB|EVAL|PACK|실패|Error|Traceback" $L/prism_fig.log 2>/dev/null | tail -6
last=$(grep -E "^\[ep" $L/prism_fig_train.log 2>/dev/null | tail -1)
if [ -n "$last" ] && ! grep -q TRAIN_DONE $L/prism_fig_train.log; then
  read cur max el <<< $(echo "$last" | sed -E 's/.*step ([0-9]+)\/([0-9]+).* ([0-9]+)s$/\1 \2 \3/')
  rem=$(( el * (max - cur) / (cur > 0 ? cur : 1) ))
  echo "학습 $cur / $max 스텝 ($((100 * cur / max))%) · 경과 $(fmt $el) · $(awk "BEGIN{printf \"%.3f\", $el/$cur}") s/step"
  echo "학습 남은 시간 $(fmt $rem) · 전체 남은 시간 약 $(fmt $((rem + AFTER))) (GLB·평가 1시간 추정 포함)"
  grep -E "^\[val\]" $L/prism_fig_train.log | tail -1 | cut -c1-160
elif grep -q TRAIN_DONE $L/prism_fig_train.log 2>/dev/null; then
  echo "학습 끝 · GLB/평가 진행 중 (로그 위 줄 참고)"
else
  echo "학습 로그 없음 (아직 시작 전)"
fi
