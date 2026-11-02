#!/bin/bash
# PRISM 학습. 사용: STEPS=20000 BS=8 [TAG=_a] bash train.sh [--resume]   로그 logs/prism_shape$TAG.log
source $(dirname $0)/common.sh || exit 1
cd $W/run && nohup python $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --init_ckpt $INIT \
  --out_dir $W/prism_shape$TAG --bs ${BS:-8} --max_step ${STEPS:-20000} ${EXTRA} $1 > $W/logs/prism_shape$TAG.log 2>&1 < /dev/null & disown
echo "started; 로그 logs/prism_shape$TAG.log"
