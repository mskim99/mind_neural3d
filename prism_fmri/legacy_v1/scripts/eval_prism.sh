#!/bin/bash
# PRISM 4단계: TRELLIS 메시(trellis/<이름>/<이름>.glb) 평가 — MinD-3D++ 평가와 같은 렌더(render_eval.py = 버그 수정판)·지표(eval_metrics.py)
# 평가 GT 렌더는 이미 있는 $W/eval_obj_bs28/render/gt_obj 재사용 (공식 render_mesh 설정, 432개)
# 사용법: MESH=<…/trellis 폴더> NAME=<결과 이름> bash eval_prism.sh
# 주의: TRELLIS는 카메라를 받지 않아 출력 메시의 방위가 GT와 맞는다는 보장이 없음 → 뷰별 지표(LPIPS·SSIM·PSNR) 해석 전에 방위 확인
source $(dirname $0)/common.sh || exit 1
B=$W/tools/blender-4.2.3-linux-x64/blender; E=$W/eval_prism/${NAME:?NAME 필요}; mkdir -p $E/lists
export BLENDER_USER_RESOURCES=$W/cache/blender CUDA_CACHE_MAXSIZE=4294967296
find ${MESH:?MESH 필요} -mindepth 2 -maxdepth 2 -name "*.glb" | sort > $E/lists/pred.txt; ls $W/eval_gt/gt_obj_glbs/*.glb > $E/lists/gt.txt; wc -l $E/lists/*.txt
$B -b -P $W/scripts/render_eval.py -- --list $E/lists/pred.txt --out_root $E/render_pred 2>&1 | grep -E "FAIL|DONE"
$W/venv/bin/python $W/scripts/eval_metrics.py --pred_render $E/render_pred --gt_render $W/eval_obj_bs28/render/gt_obj \
  --pred_mesh_list $E/lists/pred.txt --gt_mesh_list $E/lists/gt.txt --out $E/metrics.json 2>&1 | grep -viE "warn"
echo EVAL_PRISM_DONE
