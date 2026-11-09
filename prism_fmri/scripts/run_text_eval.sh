#!/bin/bash
# TRELLIS 텍스트판 외형 104 평가 (11-09, Pro6000 GPU 1, 추론만): 형상(SS) = fig 예측 토큰, 외형(SLat) = 텍스트판 "a {예측 범주}"
#  sub-0001 (text_s1) · sub-0008 (text_s8) test 104 → GLB → 8지표 → 색 일치 → 정성 5물체 → text_eval.tar (TEXT_EVAL_DONE)
#  비교: fig · slat_retr · slat_gt (sub-0001, 이미 평가) / fig_s8 · slat_gt_s8 (run_sub8_base.sh)
#  sub-0008 예측은 run_sub8_base.sh 1단계 결과를 씀 → 그 스크립트의 GLB 단계가 끝난 뒤 실행
# 사용: nohup bash run_text_eval.sh > $W/logs/prism_text_eval.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
PY=$W/venv/bin/python; O=$W/prism_text; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}
declare -A INF=([text_s1]=$W/prism_renorm/fig/infer/sub-0001_keep [text_s8]=$W/prism_sub8/infer/sub-0008_keep)
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $O
for n in text_s1 text_s8; do [ -f ${INF[$n]}/components.npz ] || { echo "예측 없음 ${INF[$n]}"; exit 1; }; done

step "1 문장 · GLB 104 → 8지표"
EV=""
for n in text_s1 text_s8; do
  I=${INF[$n]}; g=$O/glb/$n
  $PY $PR/text_prompts.py --comp $I/components.npz --out $O/prompt_$n.json
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/tokens --slat_text $O/prompt_$n.json --text_model $TM --out $g | grep -E "텍스트판|GEN_DONE|Error|없음"
  echo "GLB $n $(ls $g/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || { echo "GLB $n 104개 미만 → 멈춤"; exit 1; }
  [ -n "$EV" ] && wait $EV
  (cd $SPP_ROOT && bash run_spp.sh $n $g glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_$n.log 2>&1; echo "EVAL $n 끝 | $(tail -2 $W/logs/spp_$n.log | head -1)") &
  EV=$!
done
wait $EV

step "2 색 일치 · 정성 · 묶기"
P=$O/pack; mkdir -p $P
R() { echo $1=$SPP_ROOT/out/$1/render_a_L16000; }
$PY $PR/scripts/color_agree.py --npz $PR/data/stim_color.npz --ids $DR/annotations/core_test_list.txt --train_ids $DR/annotations/core_train_list.txt \
  --json $P/color_agree.json $(R fig) $(R slat_retr) $(R slat_gt) $(R text_s1) $(R fig_s8) $(R slat_gt_s8) $(R text_s8) | tee $P/color_agree.txt
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for n in text_s1 text_s8; do
  D=$SPP_ROOT/out/$n; cp $D/values.txt $P/values_$n.txt 2>/dev/null || echo "지표 없음 $n"; cp $O/prompt_$n.json $P/
  for u in $UIDS; do
    f=$(find $D/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
    [ -n "$f" ] && mkdir -p $P/qual/$n/$u && cp $f/{0,2,4}.png $P/qual/$n/$u/
  done
done
cd $O && tar cf text_eval.tar pack && echo "PACK $(find pack -name '*.png' | wc -l)개 (예상 30)"
echo "=== 결과"; for x in fig slat_retr slat_gt text_s1 fig_s8 slat_gt_s8 text_s8; do echo "$x $(cat $SPP_ROOT/out/$x/values.txt 2>/dev/null)"; done
step "TEXT_EVAL_DONE"
