#!/bin/bash
# vshape_g0 + TRELLIS 텍스트판 외형 (11-10, Pro6000 GPU 1, 추론만): 형상(SS) = vshape_g0 예측 토큰, 외형(SLat) = 텍스트판 "a {예측 범주}"
#  sub-0001 (vt_s1, 예측은 run_vshape.sh 결과) · sub-0008 (vt_s8, 여기서 추론) test 104 → GLB → 8지표(이전 설정, 임시) → 색 일치 → 정성 5물체
#  비교: text_s1 · text_s8 (fig + 텍스트판, run_text_eval.sh) / vs_vshape_g0 (vshape_g0 이미지판)
#  → vshape_text.tar (VSHAPE_TEXT_DONE)
# 사용: nohup bash run_vshape_text.sh > $W/logs/prism_vshape_text.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
PY=$W/venv/bin/python; O=$W/prism_vshape/text; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}; CK=$W/prism_vshape/vshape_g0/model_final.pt
I8=$W/prism_vshape/infer/vshape_g0
declare -A INF=([vt_s1]=$I8/sub-0001_keep [vt_s8]=$I8/sub-0008_keep)
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $O
[ -f $CK ] || { echo "체크포인트 없음 $CK"; exit 1; }

step "0 sub-0008 추론"
[ -f ${INF[vt_s8]}/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I8 --subs sub-0008 --perceptive keep | grep -E "범주|DONE|Error"
for n in vt_s1 vt_s8; do [ -f ${INF[$n]}/components.npz ] || { echo "예측 없음 ${INF[$n]}"; exit 1; }; done

step "1 문장 · GLB 104 → 8지표"
EV=""
for n in vt_s1 vt_s8; do
  I=${INF[$n]}; g=$O/glb/$n
  $PY $PR/text_prompts.py --comp $I/components.npz --out $O/prompt_$n.json
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/tokens --slat_text $O/prompt_$n.json --text_model $TM --out $g | grep -E "텍스트판|GEN_DONE|Error|없음"
  echo "GLB $n $(ls $g/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || { echo "GLB $n 104개 미만 → 멈춤"; exit 1; }
  [ -n "$EV" ] && wait $EV
  (cd $SPP_ROOT && bash run_spp.sh $n $g glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_$n.log 2>&1; echo "EVAL $n 끝 | $(grep -E '^2way' $W/logs/spp_$n.log | tail -1)") &
  EV=$!
done
wait $EV

step "2 색 일치 · 정성 · 묶기"
P=$O/pack; mkdir -p $P
R() { echo $1=$SPP_ROOT/out/$1/render_a_L16000; }
$PY $PR/scripts/color_agree.py --npz $PR/data/stim_color.npz --ids $DR/annotations/core_test_list.txt --train_ids $DR/annotations/core_train_list.txt \
  --json $P/color_agree.json $(R text_s1) $(R vs_vshape_g0) $(R vt_s1) $(R text_s8) $(R vt_s8) | tee $P/color_agree.txt
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for n in vt_s1 vt_s8; do
  D=$SPP_ROOT/out/$n; cp $D/values.txt $P/values_$n.txt 2>/dev/null || echo "지표 없음 $n"; cp $O/prompt_$n.json $P/
  cp ${INF[$n]}/components.npz $P/components_$n.npz 2>/dev/null
  for u in $UIDS; do
    f=$(find $D/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
    [ -n "$f" ] && mkdir -p $P/qual/$n/$u && cp $f/{0,2,4}.png $P/qual/$n/$u/
  done
done
cd $O && tar cf vshape_text.tar pack && echo "PACK $(find pack -name '*.png' | wc -l)개 (예상 30)"
echo "=== 결과"; for x in text_s1 vs_vshape_g0 vt_s1 text_s8 vt_s8; do echo "$x $(cat $SPP_ROOT/out/$x/values.txt 2>/dev/null)"; done
step "VSHAPE_TEXT_DONE"
