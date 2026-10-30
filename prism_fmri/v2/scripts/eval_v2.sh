#!/bin/bash
# PRISM v2 Shape test 104 평가: TRELLIS GLB → render_eval.py(f3b919ad) → eval_metrics.py
#  평가 GT = 4090에서 MinD-3D++ Shape 재현(공개 가중치 10-way 0.628)에 쓴 것 그대로 옮겨 옴
#   $W/eval_gt_shape/gt_shape/<uid>/{0..5}.png (렌더) + $W/eval_gt_shape/ShapeNetCore_sub01/<syn>/<uid>/models/model_normalized.obj (메시)
# 사용: [TAG=] [SUB=sub-0001] [PERC=keep|mean] [M0=1] bash eval_v2.sh  → $W/prism_v2_infer$TAG/<sub>_<perc>/metrics$SFX.json, 로그 logs/prism_v2_eval$TAG$SFX.log
#  M0=1: GLB 재질 metallic 0으로 렌더 (render_eval_m0.py, make_render_m0.py로 생성) → render_m0/, metrics_m0.json. 기존 render/·metrics.json은 그대로
source $(dirname $0)/common_v2.sh || exit 1
SUB=${SUB:-sub-0001}; PC=${PERC:-keep}; R=$W/prism_v2_infer$TAG/${SUB}_$PC; G=$W/eval_gt_shape
B=$W/tools/blender-4.2.3-linux-x64/blender
RS=$W/scripts/render_eval.py; SFX=
if [ "$M0" = 1 ]; then
  SFX=_m0; RS=$W/scripts/render_eval_m0.py
  $W/venv/bin/python $(dirname $0)/make_render_m0.py $W/scripts/render_eval.py $RS || exit 1
fi
export BLENDER_USER_RESOURCES=$W/cache/blender CUDA_CACHE_MAXSIZE=4294967296
[ -d $G/gt_shape ] || { echo "평가 GT 없음: $G"; exit 1; }
mkdir -p $R/lists
sed 's/\r$//' $DR/annotations/core_test_list.txt | awk -v g=$G/ShapeNetCore_sub01 'NF{print g"/"$0"/models/model_normalized.obj"}' > $R/lists/gt.txt
find $R/glb -mindepth 2 -maxdepth 2 -name "*.glb" | sort > $R/lists/pred.txt
wc -l $R/lists/gt.txt $R/lists/pred.txt
run(){
  echo "== 렌더 $(date +%T)"
  $B -b -P $RS -- --list $R/lists/pred.txt --out_root $R/render$SFX 2>&1 | grep -E "FAIL|DONE"
  echo "   $(ls $R/render$SFX | wc -l) dirs"
  echo "== 지표 $(date +%T)"
  $W/venv/bin/python $W/scripts/eval_metrics.py --pred_render $R/render$SFX --gt_render $G/gt_shape \
    --pred_mesh_list $R/lists/pred.txt --gt_mesh_list $R/lists/gt.txt --out $R/metrics$SFX.json 2>&1 | grep -viE "warn" | tail -n 4
  echo PRISM_V2_EVAL_DONE $(date +%T)
}
nohup bash -c "$(declare -f run); W=$W R=$R G=$G B=$B RS=$RS SFX=$SFX; export CUDA_VISIBLE_DEVICES BLENDER_USER_RESOURCES CUDA_CACHE_MAXSIZE; run" > $W/logs/prism_v2_eval$TAG$SFX.log 2>&1 < /dev/null & disown
echo "started → $R/metrics$SFX.json, 로그 logs/prism_v2_eval$TAG$SFX.log"
