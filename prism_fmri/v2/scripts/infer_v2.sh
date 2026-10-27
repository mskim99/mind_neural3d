#!/bin/bash
# PRISM v2 추론: test fMRI → 조건 토큰 (MinD-3D++ venv) → TRELLIS GLB (meshgen venv, mg_guard로 캐시 맞춤). 사용: [TAG=_a] [PERC=keep|mean] bash infer_v2.sh
#   로그 logs/prism_v2_infer$TAG.log (토큰) / logs/prism_v2_gen$TAG.log (GLB)
source $(dirname $0)/common_v2.sh || exit 1
CK=$W/prism_v2_shape$TAG/model_final.pt; OUT=$W/prism_v2_infer$TAG; PC=${PERC:-keep}
CMD="cd $W/run && python $V2/infer_prism_v2.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $OUT --perceptive $PC > $W/logs/prism_v2_infer$TAG.log 2>&1"
for s in ${SUBS//,/ }; do
  CMD="$CMD && (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES && python $V2/trellis_tools.py gen --pred $OUT/${s}_$PC/tokens --out $OUT/${s}_$PC/glb) >> $W/logs/prism_v2_gen$TAG.log 2>&1"
done
nohup bash -c "$CMD" < /dev/null > /dev/null 2>&1 & disown
echo "started → $OUT"
