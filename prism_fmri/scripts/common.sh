#!/bin/bash
# PRISM 공통 경로 (source로 불러 씀). 모두 /home/oem/jhj 안
source /home/oem/jhj/MinD-3D++/jhj_guard.sh || exit 1
W=/home/oem/jhj/MinD-3D++; MG=/home/oem/jhj/meshgen_tools
PR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # 11-17: 이 스크립트가 있는 작업 폴더의 코드 (본 폴더면 $W/mind_neural3d/prism_fmri 그대로, 두 번째 작업 폴더 mind_neural3d_p면 그쪽)
DR=$W/data/fMRI-Shape                       # npy_data(sub-0001, sub-0002) + annotations
TGT=$W/data/shape_targets                   # targets/ frames/ dino/ (Slot3D와 같이 쓰는 정답 폴더)
INIT=$W/weights/MinD-3D-pp/fMRI-Shape/sub01_encoder.pt   # MinD-3D++ Shape 공개 가중치 sub01의 fMRI 인코더만
SUBS=${SUBS:-sub-0001,sub-0002}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
