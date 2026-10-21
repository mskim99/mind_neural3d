#!/bin/bash
# PRISM fMRI판 Pro6000 공통 경로 (source로 불러 씀). 모두 /home/oem/jhj/MinD-3D++ 안.
source /home/oem/jhj/MinD-3D++/jhj_guard.sh || exit 1
W=/home/oem/jhj/MinD-3D++
P=$W/mind_neural3d/prism_fmri                 # 이 저장소의 prism_fmri 폴더 (git clone 위치)
D=${DATASET:-obj}
DR=$W/data/fMRI-Objaverse; IR=$W/data/GTviews/obj; LIST=$DR/test_list.txt; FD=$DR/${SUB:-sub-0001}/npy_data
[ "$D" = shape ] && { DR=$W/data/fMRI-Shape; IR=$W/data/GTviews/shape; LIST=$DR/annotations/core_test_list.txt; FD=$DR/${SUB:-sub-0001}; }
INIT=${INIT:-$W/weights/MinD-3D-pp/fMRI-Objaverse/sub01.pt}   # MinD-3D++ 공개 가중치 (HF Fudan-fMRI/MinD-3D-pp)
export CUDA_VISIBLE_DEVICES=${GPU:-1}
