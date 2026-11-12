#!/bin/bash
# 모델 대비 뇌 지도용: 기준 모델(vshape_g0)의 성분별 입력 기여 지도를 물체별로 저장 (7명, test 104) — 총괄 인계 3-1
#  → $W/prism_attr/obj_g0/attr_obj_<sub>.npz (S · V · P, 물체마다 합 1 × 65536, float16) + attr_<sub>.npz, 묶음 obj_g0.tar (ATTR_OBJ_DONE)
#  검정(성분 대비 S−V · S−P · V−P, 물체 간 부호 순열)과 그림은 로컬에서
# 사용: GPU=1 nohup bash run_attr_obj.sh > $W/logs/attr_obj.log 2>&1 < /dev/null & disown
SUBS=${SUBS:-sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008}
source $(dirname $0)/common.sh || exit 1
CK=${CK:-$W/prism_vshape/vshape_g0/model_final.pt}; O=$W/prism_attr/obj_g0; mkdir -p $O; cd $W/run
[ -f $CK ] || { echo "없음: $CK"; exit 1; }
for s in ${SUBS//,/ }; do
  [ -f $O/attr_obj_$s.npz ] && { echo "이미 있음 $s"; continue; }
  echo "=== $s $(date +%H:%M:%S)"
  $W/venv/bin/python $PR/prism_attr.py --pr $PR --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $O --subs $s --per_obj 1 2>&1 | grep -E "^sub-.*(/104|개,)|Error|Traceback" | tail -3
done
cd $W/prism_attr && tar cf obj_g0.tar obj_g0 && echo "PACK $(ls obj_g0/attr_obj_*.npz | wc -l)개 (예상 7) $(du -sh obj_g0.tar | cut -f1) md5 $(md5sum obj_g0.tar | cut -c1-8)"
echo "=== ATTR_OBJ_DONE $(date +%H:%M:%S)"
