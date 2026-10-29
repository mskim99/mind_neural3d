#!/bin/bash
# MinD-3D++ Objaverse 재학습 (학습 GT = 자극 영상 프레임, train_obj_vid 90k) test 432 평가
#  6뷰: 우리 재학습 model_final.pt + 우리 LoRA(r64) 병합 (배치 28 평가와 같음 — 공개 가중치 LoRA 문제와 무관)
#  3D : InstantMesh 두 카메라 — video(영상 자세 보정, 10-26 cam_calib) / v12(공식) → 비교용
#  렌더 render_eval.py(f3b919ad) + 지표 eval_metrics.py, 평가 GT 렌더는 eval_obj_bs28/render/gt_obj 재사용
# 사용법: bash eval_vidgt.sh   → $W/eval_vidgt/metrics_{video,v12}.json, 로그 logs/eval_vidgt.log
source $(dirname $0)/common.sh || exit 1
O=$W/eval_vidgt; CK=${CK:-$W/train_obj_vid/model_final.pt}; mkdir -p $O/lists
B=$W/tools/blender-4.2.3-linux-x64/blender
export BLENDER_USER_RESOURCES=$W/cache/blender CUDA_CACHE_MAXSIZE=4294967296
run(){
  echo "== 6뷰 $(date +%T)"
  cd $W/MinD-3D/InstantMesh && python $P/infer_prism_fmri.py ../configs/mind3d_pp_infer.yaml --dataset obj --list $LIST \
    --fmri_dir $FD --unet_path $CK --merge_lora --z123 $W/weights/zero123plus-v1.2 --out_dir $O 2>&1 | grep -E "^\[mv|ALL_DONE|missing|merge|Error"
  for c in video v12; do
    echo "== InstantMesh $c $(date +%T)"
    python $P/tools/instantmesh_baseline.py ../configs/mind3d_pp_infer.yaml --lrm_ckpt $W/weights/InstantMesh/instant_mesh_large.ckpt \
      --out_dir $O --cam $c --mesh_dir meshes_$c 2>&1 | grep -E "^\[lrm|ALL_DONE|Error"
    ls $O/meshes_$c/*.obj > $O/lists/pred_$c.txt
    for f in $(cat $O/lists/pred_$c.txt); do n=$(basename $f .obj); echo $W/eval_gt/gt_obj_glbs/$n.glb; done > $O/lists/gt_$c.txt
    echo "== 렌더 $c $(date +%T)"
    $B -b -P $W/scripts/render_eval.py -- --list $O/lists/pred_$c.txt --out_root $O/render_$c --obj_up Z 2>&1 | grep -E "FAIL|DONE"
    echo "== 지표 $c $(date +%T)"
    $W/venv/bin/python $W/scripts/eval_metrics.py --pred_render $O/render_$c --gt_render $W/eval_obj_bs28/render/gt_obj \
      --pred_mesh_list $O/lists/pred_$c.txt --gt_mesh_list $O/lists/gt_$c.txt --out $O/metrics_$c.json 2>&1 | grep -viE "warn" | tail -n 3
  done
  echo EVAL_VIDGT_DONE $(date +%T)
}
nohup bash -c "$(declare -f run); W=$W P=$P FD=$FD LIST=$LIST CK=$CK O=$O B=$B; export CUDA_VISIBLE_DEVICES BLENDER_USER_RESOURCES CUDA_CACHE_MAXSIZE; run" > $W/logs/eval_vidgt.log 2>&1 < /dev/null & disown
echo "started → $O, 로그 logs/eval_vidgt.log"
