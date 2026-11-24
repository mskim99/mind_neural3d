#!/bin/bash
# P-3 (11-24, 5090): 최종 Ours 7명 모델에서 학습 피험자끼리 선형 변환(train 1,300개 짝)으로 서로의 갈래에 넣어 범주 점수를 평균 — 범주 정확도만 (재학습 · 3D 없음)
# 사용 (절대 경로로): nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_p3.sh > $W/logs/p3.log 2>&1 < /dev/null &
GPU=0 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
cd $W/run; PY=$W/venv/bin/python; O=$W/prism_p3; mkdir -p $O
S7=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
if [ ! -f $O/DONE ]; then wait_gpu_free p3; echo "=== p3 $(date +%H:%M:%S)"
  $PY $PR/align_infer.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_full/ausvp/model_final.pt --subs $S7 --fmri_subs $S7 --space feat --fmap full --fit mean --ns 1300 --save 1300,own --save_tok 0 --out $O && touch $O/DONE; fi
$PY $PR/p3_ens.py $O
echo "=== P3_DONE $(date +%H:%M:%S)"
