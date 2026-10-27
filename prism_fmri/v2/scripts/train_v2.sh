#!/bin/bash
# PRISM v2 학습. 사용: STEPS=20000 BS=8 [TAG=_a] bash train_v2.sh [--resume]   로그 logs/prism_v2_shape$TAG.log
source $(dirname $0)/common_v2.sh || exit 1
cd $W/run && nohup python $V2/train_prism_v2.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --init_ckpt $INIT \
  --out_dir $W/prism_v2_shape$TAG --bs ${BS:-8} --max_step ${STEPS:-20000} ${EXTRA} $1 > $W/logs/prism_v2_shape$TAG.log 2>&1 < /dev/null & disown
echo "started; 로그 logs/prism_v2_shape$TAG.log"
