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
# 11-19 Pro6000(논리 코어 72 = 물리 36 × 하이퍼스레딩 2, 짝 = N · N+36) 공용 서버 배려 (사용자: CPU는 러프하게 절반만):
#  이 셸과 여기서 띄우는 모든 자식을 물리 코어 18~35(논리 18-35,54-71)에만 올림 + CPU 우선순위 낮게(nice 10) + 디스크 우선순위 최하 + 프로세스당 스레드 8
#  5090(24코어) 등 다른 서버에는 안 걸림. 끄기: CPUSET=none, 범위 바꾸기: CPUSET=…
if [ "$(nproc --all)" = 72 ] && [ -z "${JHJ_PINNED:-}" ] && [ "${CPUSET:-}" != none ]; then
  CPUSET=${CPUSET:-18-35,54-71}
  taskset -cp $CPUSET $$ > /dev/null && renice -n 10 -p $$ > /dev/null; ionice -c2 -n7 -p $$ 2> /dev/null
  export JHJ_PINNED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
  echo "[pin] CPU $CPUSET (물리 코어 절반) · nice 10 · 디스크 우선순위 최하 · 스레드 8"
fi
