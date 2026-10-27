#!/bin/bash
# PRISM 1단계: 데이터 확인 + 20스텝 시험 (모델 로드·공개 가중치 불러오기·손실·메모리). 사용법: bash smoke_prism.sh   (BS=2 AC=4 기본)
source $(dirname $0)/common.sh || exit 1
cd $W/run
torchrun --nproc_per_node=1 --master_port=29571 $P/train_prism_fmri.py --dataset $D --repo $W/MinD-3D --data_root $DR --image_root $IR \
  --z123 $W/weights/zero123plus-v1.2 --out_dir $W/prism_smoke --check_data 2 2>&1 | grep -v -i warn
torchrun --nproc_per_node=1 --master_port=29571 $P/train_prism_fmri.py --dataset $D --repo $W/MinD-3D --data_root $DR --image_root $IR \
  --z123 $W/weights/zero123plus-v1.2 --init_ckpt $INIT --out_dir $W/prism_smoke --bs ${BS:-2} --accum ${AC:-4} --grad_ckpt $FRZ \
  --max_step 20 --print_every 5 --save_every 0 2>&1 | grep -v -i warn | tee $W/logs/prism_smoke.log \
  | grep -E "objects|init\]|trainable|step |TRAIN_DONE|Error|error|out of memory|Traceback"
