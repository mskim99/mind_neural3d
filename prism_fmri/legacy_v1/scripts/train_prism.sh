#!/bin/bash
# PRISM 2단계: 학습 (백그라운드). 사용법: STEPS=5000 BS=2 AC=4 [ORTHO=1 STAGE1=1000] [OUT_TAG=_s1o] bash train_prism.sh [--resume]   (FREEZE= 로 인코더도 학습)
source $(dirname $0)/common.sh || exit 1
cd $W/run
nohup torchrun --nproc_per_node=1 --master_port=${PORT:-$([ -n "$OUT_TAG" ] && echo 29573 || echo 29572)} $P/train_prism_fmri.py --dataset $D --repo $W/MinD-3D --data_root $DR --image_root $IR \
  --z123 $W/weights/zero123plus-v1.2 --init_ckpt $INIT --out_dir $W/prism_train_$D$OUT_TAG --bs ${BS:-2} --accum ${AC:-4} --grad_ckpt $FRZ \
  --max_step ${STEPS:-5000} --save_every 1000 --lambda_ortho ${ORTHO:-0} --stage1_steps ${STAGE1:-0} $1 > $W/logs/prism_train_$D$OUT_TAG.log 2>&1 < /dev/null & disown
echo "started; 진행: watch -n 1 'tail -n 3 $W/logs/prism_train_$D$OUT_TAG.log'"
