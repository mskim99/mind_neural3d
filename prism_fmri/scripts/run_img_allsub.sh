#!/bin/bash
# A안 2단계 7명 확인 (11-10, 5090 GPU 0, 추론만): prism_img10k vs fig 기준선 — 피험자별 test 범주 정확도, 이미지 임베딩 검색(2-way · top-5)
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; SUBS7=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
cd $W/run
for m in prism_fig prism_img10k; do
  $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $W/$m/model_final.pt --out $W/$m/infer7 --subs $SUBS7 --perceptive keep --save_tokens 0 | grep -E "범주"
done
for s in ${SUBS7//,/ }; do
  echo "$s $($PY $PR/eval_img.py --comp $W/prism_img10k/infer7/${s}_keep/components.npz --clipimg $TGT/clipimg --train_list $DR/annotations/core_train_list.txt 2>/dev/null | grep -E "pred|class_mean_pred" | tr -s " " | tr "\n" " ")"
done
echo ALLSUB_DONE
