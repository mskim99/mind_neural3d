#!/bin/bash
# PRISM 3단계: test 추론 → 6뷰(images/, views/<이름>/00~05.png) → TRELLIS 3D (trellis/<이름>/<이름>.glb). 제안 기법 3D는 TRELLIS만 (사용자 10-26)
# 사용법: CKPT=<model_final.pt> bash infer_prism.sh [--limit N]      6뷰가 이미 있으면 VIEWS_DONE=1 (TRELLIS만)
# 학습 중 체크포인트(model_latest.pt, 인코더·LoRA만)는 뒤에 --base_unet $W/weights/train_init/diffusion_pytorch_model.bin
# 로그: logs/prism_infer_<D>.log (6뷰), logs/prism_trellis_<D>.log (TRELLIS)
source $(dirname $0)/common.sh || exit 1
CK=${CKPT:-$W/prism_train_$D/model_final.pt}; OUT=${OUT:-$W/prism_infer_$D/$(basename $(dirname $CK))}
MG=/home/oem/jhj/meshgen_tools
VIEWS="cd $W/MinD-3D/InstantMesh && python $P/infer_prism_fmri.py ../configs/mind3d_pp_infer.yaml --dataset $D --list $LIST --fmri_dir $FD --unet_path $CK --merge_lora --z123 $W/weights/zero123plus-v1.2 --out_dir $OUT $* > $W/logs/prism_infer_$D.log 2>&1"
TREL="cd $MG && $MG/venv/bin/python run_trellis.py --trellis-repo repo/third_party/TRELLIS --input-dir $OUT/views --output-dir $OUT/trellis --views-per-object 6 --no-video > $W/logs/prism_trellis_$D.log 2>&1"
if [ -n "$VIEWS_DONE" ]; then CMD="$TREL"; else CMD="$VIEWS && $TREL"; fi
nohup bash -c "$CMD" < /dev/null > /dev/null 2>&1 & disown
echo "started → $OUT (6뷰 $([ -n "$VIEWS_DONE" ] && echo 건너뜀 || echo 생성) → TRELLIS); 로그 logs/prism_infer_$D.log, logs/prism_trellis_$D.log"
