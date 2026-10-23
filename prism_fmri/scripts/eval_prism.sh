#!/bin/bash
# PRISM 4단계: 추론 메시(InstantMesh .obj) 평가 — MinD-3D++ 평가와 같은 렌더(render_eval.py = 버그 수정판)·지표(eval_metrics.py)
# 평가 GT 렌더는 이미 있는 $W/eval_obj_bs28/render/gt_obj 재사용 (공식 render_mesh 설정, 432개)
# 사용법: MESH=<…/meshes 폴더> NAME=<결과 이름> bash eval_prism.sh
source $(dirname $0)/common.sh || exit 1
B=$W/tools/blender-4.2.3-linux-x64/blender; E=$W/eval_prism/${NAME:?NAME 필요}; mkdir -p $E/lists
export BLENDER_USER_RESOURCES=$W/cache/blender CUDA_CACHE_MAXSIZE=4294967296
ls ${MESH:?MESH 필요}/*.obj > $E/lists/pred.txt; ls $W/eval_gt/gt_obj_glbs/*.glb > $E/lists/gt.txt; wc -l $E/lists/*.txt
$B -b -P $W/scripts/render_eval.py -- --list $E/lists/pred.txt --out_root $E/render_pred --obj_up Z 2>&1 | grep -E "FAIL|DONE"
$W/venv/bin/python $W/scripts/eval_metrics.py --pred_render $E/render_pred --gt_render $W/eval_obj_bs28/render/gt_obj \
  --pred_mesh_list $E/lists/pred.txt --gt_mesh_list $E/lists/gt.txt --out $E/metrics.json 2>&1 | grep -viE "warn"
echo EVAL_PRISM_DONE
