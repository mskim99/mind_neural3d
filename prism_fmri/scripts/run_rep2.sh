#!/bin/bash
# A안 반복 (11-10, 5090 GPU 0): 이미지 head 1만 · 대조군 fig 1만을 한 번 더 (시드 고정 없음 = 초기값 다름) → 7명 범주 정확도 · 임베딩 검색
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; ENC=$W/weights/MinD-3D-pp/fMRI-Shape; cd $W/run
for v in img fig; do
  M=$W/prism_${v}10k_r2; mkdir -p $M; X=""; [ $v = img ] && X="--lam_img 1 --lam_img_nce 1"
  [ -f $M/model_final.pt ] || $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step 10000 --per_subject 1 \
    --enc_dir $ENC --arch fig --sem_clip front --var_heads sr --var_mode feat $X --out_dir $M > $M/train.log 2>&1
  echo "== $v r2"; $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $M/model_final.pt --out $M/infer7 --subs $SUBS --perceptive keep --save_tokens 0 | grep -oE "sub-000[0-9]|정확도 [0-9.]+" | paste - -
  if [ $v = img ]; then for s in ${SUBS//,/ }; do echo "$s $($PY $PR/eval_img.py --comp $M/infer7/${s}_keep/components.npz --clipimg $TGT/clipimg --train_list $DR/annotations/core_train_list.txt 2>/dev/null | grep -E "pred" | tr -s " " | tr "\n" " ")"; done; fi
done
echo REP2_DONE
