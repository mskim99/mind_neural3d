#!/bin/bash
# PRISM v2 전처리: 영상 프레임 0 → TRELLIS 조건 토큰 정답 (meshgen venv). 사용: bash dino_v2.sh   로그 logs/prism_v2_dino.log
source $(dirname $0)/common_v2.sh || exit 1
cd $MG && nohup $MG/venv/bin/python $V2/trellis_tools.py dino --frames $TGT/frames --out $TGT/dino --view ${VIEW:-0} \
  > $W/logs/prism_v2_dino.log 2>&1 < /dev/null & disown
echo "started; 로그 logs/prism_v2_dino.log"
