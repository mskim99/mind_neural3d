#!/bin/bash
# PRISM 3단계: test 추론 → 6뷰(images/, views/) + InstantMesh 메시. 사용법: CKPT=<model_final.pt> bash infer_prism.sh [--no_lrm] [--limit N]
# 학습 중 체크포인트(model_latest.pt, 인코더·LoRA만): CKPT=…/model_latest.pt OUT=… bash infer_prism.sh --no_lrm --base_unet $W/weights/train_init/diffusion_pytorch_model.bin
# CKPT를 공개 가중치($INIT)로 주면 헤드 없이 항등 → MinD-3D++와 같은 결과가 나와야 함 (연결 확인용)
source $(dirname $0)/common.sh || exit 1
CK=${CKPT:-$W/prism_train_$D/model_final.pt}; OUT=${OUT:-$W/prism_infer_$D/$(basename $(dirname $CK))}
cd $W/MinD-3D/InstantMesh
nohup python $P/infer_prism_fmri.py ../configs/mind3d_pp_infer.yaml --dataset $D --list $LIST --fmri_dir $FD --unet_path $CK --merge_lora \
  --z123 $W/weights/zero123plus-v1.2 --lrm_ckpt $W/weights/InstantMesh/instant_mesh_large.ckpt --out_dir $OUT "$@" \
  > $W/logs/prism_infer_$D.log 2>&1 < /dev/null & disown
echo "started → $OUT; 진행: watch -n 1 'tail -n 3 $W/logs/prism_infer_$D.log'"
