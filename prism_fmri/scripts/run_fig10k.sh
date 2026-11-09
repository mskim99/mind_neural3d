#!/bin/bash
# A안 대조군 (11-10, 5090 GPU 0): 이미지 head 없이 fig를 1만 스텝 일정으로 → prism_img10k와 학습 길이만 같게, 7명 범주 정확도 비교
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; M=$W/prism_fig10k; cd $W/run; mkdir -p $M
[ -f $M/model_final.pt ] || $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step 10000 --per_subject 1 \
  --enc_dir $W/weights/MinD-3D-pp/fMRI-Shape --arch fig --sem_clip front --var_heads sr --var_mode feat --out_dir $M > $M/train.log 2>&1
grep -E "^\[val\]" $M/train.log | tail -1
$PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $M/model_final.pt --out $M/infer7 --subs $SUBS --perceptive keep --save_tokens 0 | grep -E "범주"
echo FIG10K_DONE
