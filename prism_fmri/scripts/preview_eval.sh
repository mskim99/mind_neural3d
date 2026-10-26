#!/bin/bash
# PRISM 미리보기 평가 (Sub-1, Objaverse): TRELLIS가 끝난 물체 N개만 골라 기준점과 같은 물체로 비교
#  기준점 = MinD-3D++ 공개 가중치 sub01 + 우리 추론, **공개 LoRA 미적용**(공식 로더와 같음) (6뷰 → InstantMesh 공식 v1.2 카메라) — 4090에서 10-way 0.637(논문 0.636)이 나온 방식
#  PRISM  = 6뷰 → TRELLIS GLB (이미 만든 것)
#  둘 다 render_eval.py(버그 수정판)로 렌더 → eval_metrics.py(공식 정의). 평가 GT 렌더는 eval_obj_bs28/render/gt_obj 재사용
# 사용법: N=50 bash preview_eval.sh   → 결과 $W/preview_obj/metrics_{base_nolora,prism}.json (10-26 첫 실행의 metrics_base.json은 공개 LoRA 병합판 — 틀린 기준점), 로그 logs/preview_obj.log
source $(dirname $0)/common.sh || exit 1
N=${N:-50}; PV=$W/preview_obj; PR=$W/prism_infer_obj/prism_train_obj; mkdir -p $PV/lists
B=$W/tools/blender-4.2.3-linux-x64/blender
export BLENDER_USER_RESOURCES=$W/cache/blender CUDA_CACHE_MAXSIZE=4294967296
# TRELLIS가 끝난 물체 앞 N개 (이름 정렬). 목록이 이미 있으면 그대로 씀 (같은 물체로 다시 비교)
[ -s $PV/lists/ids.txt ] || find $PR/trellis -mindepth 2 -maxdepth 2 -name "*.glb" | sort | head -n $N > $PV/lists/prism.txt
[ -s $PV/lists/ids.txt ] || sed 's#.*/\([0-9a-f]*\)\.glb$#\1#' $PV/lists/prism.txt > $PV/lists/ids.txt
{ head -n 1 $LIST; grep -F -f $PV/lists/ids.txt $LIST; } > $PV/lists/test_sub.txt
for k in $(cat $PV/lists/ids.txt); do echo $W/eval_gt/gt_obj_glbs/$k.glb; done > $PV/lists/gt.txt
wc -l $PV/lists/*.txt
run(){
  echo "== base 6뷰 $(date +%T)"
  cd $W/MinD-3D/InstantMesh && python $P/infer_prism_fmri.py ../configs/mind3d_pp_infer.yaml --dataset $D --list $PV/lists/test_sub.txt \
    --fmri_dir $FD --unet_path $INIT --z123 $W/weights/zero123plus-v1.2 --out_dir $PV/base_nolora 2>&1 | grep -E "^\[mv|ALL_DONE|missing|Error"
  echo "== base InstantMesh (v12) $(date +%T)"
  python $P/tools/instantmesh_baseline.py ../configs/mind3d_pp_infer.yaml --lrm_ckpt $W/weights/InstantMesh/instant_mesh_large.ckpt \
    --out_dir $PV/base_nolora --cam v12 2>&1 | grep -E "^\[lrm|ALL_DONE|Error"
  ls $PV/base_nolora/meshes/*.obj > $PV/lists/base.txt
  echo "== 렌더 $(date +%T)"
  $B -b -P $W/scripts/render_eval.py -- --list $PV/lists/base.txt --out_root $PV/render_base_nolora --obj_up Z 2>&1 | grep -E "FAIL|DONE"
  $B -b -P $W/scripts/render_eval.py -- --list $PV/lists/prism.txt --out_root $PV/render_prism 2>&1 | grep -E "FAIL|DONE"
  echo "== 지표 $(date +%T)"
  for t in base_nolora prism; do
    [ -s $PV/metrics_${t}.json ] && continue
    $W/venv/bin/python $W/scripts/eval_metrics.py --pred_render $PV/render_$t --gt_render $W/eval_obj_bs28/render/gt_obj \
      --pred_mesh_list $PV/lists/${t%_nolora}.txt --gt_mesh_list $PV/lists/gt.txt --out $PV/metrics_${t}.json 2>&1 | grep -viE "warn" | tail -n 2
  done
  echo PREVIEW_DONE $(date +%T)
}
nohup bash -c "$(declare -f run); W=$W P=$P D=$D FD=$FD INIT=$INIT PV=$PV B=$B; export CUDA_VISIBLE_DEVICES BLENDER_USER_RESOURCES CUDA_CACHE_MAXSIZE; run" > $W/logs/preview_obj.log 2>&1 < /dev/null & disown
echo "started → $PV, 로그 logs/preview_obj.log"
