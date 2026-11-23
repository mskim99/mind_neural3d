#!/bin/bash
# 11-23: 7명 모델(prism_full/ausvp)에 처음 보는 4번 피험자를 같은 물체 짝 N개로 붙이기 — 짝 수 0 / 50 / 100 / 300 / 1300, 범주 정확도만 (재학습 · 3D 생성 없음)
#  feat_full      : 기준 (ViT 토큰 전체를 한 벡터로 본 선형 변환, 학습 피험자 한 명에게)
#  feat_full_pool : 목표를 "그 피험자 + 다른 학습 피험자 6명을 그 피험자로 옮긴 값"의 평균으로 (목표 잡음 줄이기)
#  input_full     : 모델 입력(평면 지도 픽셀)에서 전체 선형 변환 — input_region의 비교 기준
#  input_region   : 모델 입력에서 뇌 영역마다 따로 선형 변환 (구할 값 줄이기)
#  + 설정마다 7명에게 각각 옮긴 범주 점수의 평균(align_ens.py)
# 사용 (절대 경로로): bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_align7.sh
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd $W/run; PY=$W/venv/bin/python; CK=$W/prism_full/ausvp/model_final.pt; O=$W/prism_align7; mkdir -p $O
S7=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008; NS=${NS:-0,50,100,300,1300}
run() { local n=$1; shift; [ -f $O/$n/DONE ] && { echo "[$n] 이미 끝남"; return; }; wait_gpu_free $n; echo "=== $n $(date +%H:%M:%S)"
  $PY $PR/align_infer.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --subs $S7 --fmri_subs sub-0004 --ns $NS --save $NS --save_tok 0 --fit mean --out $O/$n "$@" && touch $O/$n/DONE; }
run feat_full --space feat --fmap full
run input_full --space input --fmap full
run input_region --space input --fmap region
run feat_full_pool --space feat --fmap full --tgt_pool 1
$PY $PR/align_ens.py $O
echo "=== ALIGN7_DONE $(date +%H:%M:%S)"
