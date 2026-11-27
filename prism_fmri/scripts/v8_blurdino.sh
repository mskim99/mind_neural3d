#!/bin/bash
# V-8 ㉣ 준비: 흐릿한 이미지(정답 또는 예측) → TRELLIS DINOv2 조건 토큰. SRC=gt|pred
GPU=0 source /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/common.sh || exit 1
SRC=${SRC:-gt}; O=$W/prism_v8/blur_$SRC; cd $W/run
X=; [ $SRC = pred ] && X="--pred $W/prism_v9/pred_sub-0001.npz --base $W/prism_full/eval_ausvp/text/s01/prompt.json"
$W/venv/bin/python $PR/v8_blur.py --v9 $W/prism_v9/v9_feat.npz --ann $DR/annotations --src $SRC $X --out $O/frames
until [ "$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 9000 ]; do sleep 20; done
(cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py dino --frames $O/frames --out $O/dino --view 0 --thr 245) 2>&1 | grep -E "DINO_DONE|cond shape|Error"
echo "BLURDINO_DONE $SRC $(ls $O/dino/*.npy | wc -l) $(date +%H:%M:%S)"
