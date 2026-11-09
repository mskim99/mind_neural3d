#!/bin/bash
# sub-0008 기준값 (11-09, Pro6000 GPU 1, 추론만): 색 정보가 있는 피험자(원 신호 해독 · 색 유도 모의에서 유의)의 test 104 기준
#  fig_s8     : fig 모델 sub-0008 예측 토큰을 두 단계(형상 SS · 외형 SLat) 모두에 씀 (지금 방법 그대로)
#  slat_gt_s8 : 형상 = sub-0008 예측, 외형 = test 정답 토큰 (상한, 방법 아님)
#  → GLB 104 → 8지표 → 색 일치(fig · slat_gt sub-0001과 같이) → 정성 5물체 → sub8_base.tar (SUB8_BASE_DONE)
#  다음 비교 대상: TRELLIS 텍스트판 외형 (범주 문장 / 색 단어 문장)
# 사용: nohup bash run_sub8_base.sh > $W/logs/prism_sub8_base.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
PY=$W/venv/bin/python; FIG=$W/prism_fig/model_final.pt; O=$W/prism_sub8; S8=sub-0008; I=$O/infer/${S8}_keep
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $O

step "1 fig 모델 sub-0008 test 예측"
[ -f $I/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $FIG --out $O/infer --subs $S8 --perceptive keep --tok_pool 1 | grep -E "범주|DONE|Error|없"
[ $(ls $I/tokens/*.npy 2>/dev/null | wc -l) -ge 104 ] || { echo "sub-0008 예측 토큰 104개 미만 (fig 학습 피험자에 sub-0008이 있는지 확인)"; exit 1; }

step "2 GLB 104 → 8지표"
EV=""
for n in fig_s8 slat_gt_s8; do
  g=$O/glb/$n; extra=""; [ $n = slat_gt_s8 ] && extra="--slat_pred $TGT/dino"
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/tokens $extra --out $g | grep -E "GEN_DONE|Error|없음"
  echo "GLB $n $(ls $g/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ -n "$EV" ] && wait $EV
  (cd $SPP_ROOT && bash run_spp.sh $n $g glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_$n.log 2>&1; echo "EVAL $n 끝 | $(tail -2 $W/logs/spp_$n.log | head -1)") &
  EV=$!
done
wait $EV

step "3 색 일치 · 정성 · 묶기"
P=$O/pack; mkdir -p $P
$PY $PR/scripts/color_agree.py --npz $PR/data/stim_color.npz --ids $DR/annotations/core_test_list.txt --train_ids $DR/annotations/core_train_list.txt \
  --json $P/color_agree.json fig_s1=$SPP_ROOT/out/fig/render_a_L16000 slat_gt_s1=$SPP_ROOT/out/slat_gt/render_a_L16000 \
  fig_s8=$SPP_ROOT/out/fig_s8/render_a_L16000 slat_gt_s8=$SPP_ROOT/out/slat_gt_s8/render_a_L16000 | tee $P/color_agree.txt
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for n in fig_s8 slat_gt_s8; do
  R=$SPP_ROOT/out/$n; cp $R/values.txt $P/values_$n.txt 2>/dev/null || echo "지표 없음 $n"
  for u in $UIDS; do
    f=$(find $R/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
    [ -n "$f" ] && mkdir -p $P/qual/$n/$u && cp $f/{0,2,4}.png $P/qual/$n/$u/
  done
done
cp $I/components.npz $P/components_s8.npz
cd $O && tar cf sub8_base.tar pack && echo "PACK $(find pack -name '*.png' | wc -l)개 (예상 30)"
echo "=== 결과"; for x in fig slat_gt fig_s8 slat_gt_s8; do echo "$x $(cat $SPP_ROOT/out/$x/values.txt 2>/dev/null)"; done
step "SUB8_BASE_DONE"
