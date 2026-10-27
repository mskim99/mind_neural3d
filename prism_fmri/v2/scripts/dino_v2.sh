#!/bin/bash
# PRISM v2 전처리: 영상 프레임 0 → TRELLIS 조건 토큰 정답 (meshgen venv). 사용: bash dino_v2.sh   로그 logs/prism_v2_dino.log
# TRELLIS 가중치·캐시는 meshgen 쪽 → meshgen 가드(mg_guard.sh)를 불러 HF·torch 캐시를 meshgen_tools/cache로 맞춤, 오프라인
source $(dirname $0)/common_v2.sh || exit 1
nohup bash -c "cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES && python $V2/trellis_tools.py dino --frames $TGT/frames --out $TGT/dino --view ${VIEW:-0}" \
  > $W/logs/prism_v2_dino.log 2>&1 < /dev/null & disown
echo "started; 로그 logs/prism_v2_dino.log"
