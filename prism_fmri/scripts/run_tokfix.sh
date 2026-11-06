#!/bin/bash
# 예측 토큰 → train 정답 토큰 공간 (11-06, Pro6000 GPU 1, 재학습 없음). 모델 prism_nop_feat, sub-0001 test 104
#  quick (기본): tokfix.py로 변형 토큰 (knn1·3·5, pca16·64·256) → 정성 14물체 × (pred · gt · 변형 6) Gaussian 렌더 + 색 → 묶기 (TOKFIX_QUICK_DONE)
#  full : FULL="knn3 pca64" 처럼 고른 변형만 GLB 104 → 8지표 → 렌더 색 · 묶기 (TOKFIX_DONE)
# 사용: nohup bash run_tokfix.sh [quick|full] > $W/logs/prism_tokfix[_full].log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
PY=$W/venv/bin/python; T=$W/prism_infer_nop_feat/sub-0001_keep/tokens; O=$W/prism_tokfix; MODE=${1:-quick}
step() { echo "=== $1 $(date +%H:%M:%S)"; }
src() { case $1 in pred) echo $T ;; gt) echo $TGT/dino ;; *) echo $O/tok/$1 ;; esac; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; [ -d $T ] || { echo "토큰 없음 $T"; exit 1; }

if [ $MODE = quick ]; then
  step "1 변형 토큰 (tokfix.py)"
  $PY $PR/tokfix.py --pred $T --tgt $TGT --data_root $DR --out $O/tok | grep -vE "^\s*$"
  step "2 빠른 확인 (정성 14물체)"
  for v in pred gt knn1 knn3 knn5 pca16 pca64 pca256; do
    trel gscheck --pred $(src $v) --out $O/quick/$v --names $PR/scripts/qual14_uids.txt | grep -E "GSCHECK_DONE|Error|없음"
    echo "  $v 채도: $($PY -c "
import json; r=[json.loads(l) for l in open('$O/quick/$v/gscheck.jsonl')]
print('Gaussian 밝기 %.1f 채도 %.1f | 텍스처 밝기 %.1f 채도 %.1f (n %d)' % tuple([sum(x[k][i] for x in r)/len(r) for k in ('gauss_mean_chroma','tex_mean_chroma') for i in (0,1)] + [len(r)]))")"
  done
  cd $O && tar cf tokfix_quick.tar quick/*/*.png quick/*/gscheck.jsonl tok/tokfix_stats.json && echo "PACK quick $(ls quick/*/*.png | wc -l)개 (예상 112)"
  step "TOKFIX_QUICK_DONE"
else
  FULL=${FULL:?FULL="knn3 pca64" 처럼 변형을 지정}
  step "3 전체 ($FULL: GLB 104 → 8지표)"
  EV=""
  for v in $FULL; do
    trel gen --pred $(src $v) --out $O/$v/glb | grep -E "GEN_DONE|Error"
    echo "GLB $v $(ls $O/$v/glb/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
    [ -n "$EV" ] && wait $EV
    (cd $SPP_ROOT && bash run_spp.sh tf_$v $O/$v/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_tf_$v.log 2>&1
     echo "EVAL $v 끝 $(date +%H:%M:%S) | $(tail -2 $W/logs/spp_tf_$v.log | head -1)") &
    EV=$!
  done
  wait
  step "4 렌더 색 · 묶기"
  RUNS="pred=$SPP_ROOT/out/nop_feat/render_a_L16000"; for v in $FULL; do RUNS="$RUNS $v=$SPP_ROOT/out/tf_$v/render_a_L16000"; done
  $PY $PR/scripts/color_stats.py --tgt $TGT --ids $DR/annotations/core_test_list.txt $RUNS | tee $O/color_stats.txt
  UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
  P=$O/pack; mkdir -p $P; cp $O/color_stats.txt $O/tok/tokfix_stats.json $P/
  for v in pred $FULL; do
    [ $v = pred ] && R=$SPP_ROOT/out/nop_feat || R=$SPP_ROOT/out/tf_$v
    mkdir -p $P/$v; cp $R/values.txt $P/$v/ 2>/dev/null || echo "지표 없음 $v"
    for u in $UIDS; do
      f=$(find $R/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
      [ -n "$f" ] && mkdir -p $P/$v/qual/$u && cp $f/{0,2,4}.png $P/$v/qual/$u/
    done
  done
  cd $O && tar cf tokfix.tar pack && echo "PACK $(find pack -name '*.png' | wc -l)개 $(du -sh tokfix.tar | cut -f1)"
  echo "=== 결과"; for v in pred $FULL; do echo "$v $(cat $P/$v/values.txt 2>/dev/null)"; done
  step "TOKFIX_DONE"
fi
