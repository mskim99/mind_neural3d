#!/bin/bash
# 색 실험 1: 예측 토큰 재조정 스윕 + 범주 지표 (11-07, Pro6000 GPU 1, 추론만 · 재학습 없음). 모델 MODEL(기본 fig), sub-0001 test 104
#  1 추론 (토큰 + 범주 점수 13개)  2 renorm.py: 변형 토큰 (ln · s137 · s160 · ln_s137 · ln_s160) + 진단 + top-3 · kNN 범주 일치
#  3 빠른 확인: 정성 14물체 × (pred · gt · 변형 5) Gaussian 렌더 + 색 → renorm_quick.tar (RENORM_QUICK_DONE)
#  4 변형 5개 GLB 104 → 8지표 (앞 변형 평가와 다음 변형 GLB를 겹쳐 돌림)  5 렌더 색 · 정성(물체 5 × 뷰 0·2·4) · 묶기 → renorm.tar (RENORM_DONE)
#  비교 기준 pred = 이미 평가한 $SPP_ROOT/out/$MODEL (run_fig.sh)
# 사용: nohup bash run_renorm.sh > $W/logs/prism_renorm.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
PY=$W/venv/bin/python; MODEL=${MODEL:-fig}; S1=sub-0001; CK=$W/prism_$MODEL/model_final.pt
O=$W/prism_renorm/$MODEL; I=$O/infer/${S1}_keep; VARS="ln s137 s160 ln_s137 ln_s160"
step() { echo "=== $1 $(date +%H:%M:%S)"; }
src() { case $1 in pred) echo $I/tokens ;; gt) echo $TGT/dino ;; *) echo $O/tok/$1 ;; esac; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; [ -f $CK ] || { echo "체크포인트 없음 $CK"; exit 1; }

step "1 추론 ($MODEL, $S1)"
[ -f $I/components.npz ] && echo "이미 있음 $I" || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $O/infer --subs $S1 --perceptive keep | grep -E "범주|DONE"
step "2 변형 토큰 · 진단 · 범주 (renorm.py)"
$PY $PR/renorm.py --pred $I/tokens --comp $I/components.npz --tgt $TGT --data_root $DR --out $O/tok | grep -E "TOK|CAT|KNN|DONE|Error"

step "3 빠른 확인 (정성 14물체)"
for v in pred gt $VARS; do
  [ -f $O/quick/$v/gscheck.jsonl ] || trel gscheck --pred $(src $v) --out $O/quick/$v --names $PR/scripts/qual14_uids.txt | grep -E "GSCHECK_DONE|Error|없음"
  echo "  $v 색: $($PY -c "
import json; r=[json.loads(l) for l in open('$O/quick/$v/gscheck.jsonl')]
print('Gaussian 밝기 %.1f 채도 %.1f | 텍스처 밝기 %.1f 채도 %.1f (n %d)' % tuple([sum(x[k][i] for x in r)/len(r) for k in ('gauss_mean_chroma','tex_mean_chroma') for i in (0,1)] + [len(r)]))")"
done
cd $O && tar cf renorm_quick.tar quick/*/*.png quick/*/gscheck.jsonl tok/renorm_stats.json && echo "PACK quick $(ls quick/*/*.png | wc -l)개 (예상 98)"
step "RENORM_QUICK_DONE"

step "4 전체 ($VARS: GLB 104 → 8지표)"
EV=""
for v in $VARS; do
  trel gen --pred $(src $v) --out $O/$v/glb | grep -E "GEN_DONE|Error"
  echo "GLB $v $(ls $O/$v/glb/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ -n "$EV" ] && wait $EV
  (cd $SPP_ROOT && bash run_spp.sh rn_${MODEL}_$v $O/$v/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_rn_${MODEL}_$v.log 2>&1
   echo "EVAL $v 끝 $(date +%H:%M:%S) | $(tail -2 $W/logs/spp_rn_${MODEL}_$v.log | head -1)") &
  EV=$!
done
wait

step "5 렌더 색 · 묶기"
ev() { [ $1 = pred ] && echo $SPP_ROOT/out/$MODEL || echo $SPP_ROOT/out/rn_${MODEL}_$1; }
RUNS=""; for v in pred $VARS; do RUNS="$RUNS $v=$(ev $v)/render_a_L16000"; done
$PY $PR/scripts/color_stats.py --tgt $TGT --ids $DR/annotations/core_test_list.txt $RUNS | tee $O/color_stats.txt
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
P=$O/pack; mkdir -p $P; cp $O/color_stats.txt $O/tok/renorm_stats.json $P/
for v in pred $VARS; do
  R=$(ev $v); mkdir -p $P/$v; cp $R/values.txt $P/$v/ 2>/dev/null || echo "지표 없음 $v"
  for u in $UIDS; do
    f=$(find $R/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
    [ -n "$f" ] && mkdir -p $P/$v/qual/$u && cp $f/{0,2,4}.png $P/$v/qual/$u/
  done
done
cd $O && tar cf renorm.tar pack && echo "PACK $(find pack -name '*.png' | wc -l)개 (예상 90) $(du -sh renorm.tar | cut -f1)"
echo "=== 결과"; for v in pred $VARS; do echo "$v $(cat $P/$v/values.txt 2>/dev/null)"; done
step "RENORM_DONE"
