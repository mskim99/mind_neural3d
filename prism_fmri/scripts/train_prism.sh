#!/bin/bash
# PRISM 2단계: 학습 (백그라운드). 사용법: STEPS=5000 bash train_prism.sh [--resume]
source $(dirname $0)/common.sh || exit 1
cd $W/run
nohup torchrun --nproc_per_node=1 --master_port=29572 $P/train_prism_fmri.py --dataset $D --repo $W/MinD-3D --data_root $DR --image_root $IR \
  --z123 $W/weights/zero123plus-v1.2 --init_ckpt $INIT --out_dir $W/prism_train_$D --bs ${BS:-7} --accum ${AC:-4} --grad_ckpt \
  --max_step ${STEPS:-5000} --save_every 1000 $1 > $W/logs/prism_train_$D.log 2>&1 < /dev/null & disown
echo "started; 진행: watch -n 1 'tail -n 3 $W/logs/prism_train_$D.log'"
