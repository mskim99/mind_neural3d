#!/bin/bash
# PRISM fMRI판 Pro6000 공통 경로 (source로 불러 씀). 모두 /home/oem/jhj/MinD-3D++ 안.
source /home/oem/jhj/MinD-3D++/jhj_guard.sh || exit 1
W=/home/oem/jhj/MinD-3D++
P=$W/mind_neural3d/prism_fmri                 # 이 저장소의 prism_fmri 폴더 (git clone 위치)
D=${DATASET:-obj}
# 학습 GT = 자극 영상 프레임 (10-23~, 학습 정답은 피험자가 본 자극 안에서만). 평가 GT·렌더는 MinD-3D++ 프로토콜 그대로
DR=$W/data/fMRI-Objaverse; IR=$W/data/GTviews_video/obj; LIST=$DR/test_list.txt; FD=$DR/${SUB:-sub-0001}/npy_data
[ "$D" = shape ] && { echo "shape 학습 GT(영상 프레임)는 아직 없음"; exit 1; }
INIT=${INIT:-$W/weights/MinD-3D-pp/fMRI-Objaverse/sub01.pt}   # MinD-3D++ 공개 가중치 (HF Fudan-fMRI/MinD-3D-pp)
FREEZE=${FREEZE-1}                                              # 기본 인코더 ViT 고정 (GPU 1 공유). 풀려면 FREEZE= (빈 값)
FRZ=${FREEZE:+--freeze_encoder}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
