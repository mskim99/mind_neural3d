#!/bin/bash
# 외형 단계 분리 (11-09, Pro6000 GPU 1, 추론만 · 재학습 없음): TRELLIS 1단계(형상, SS) = fig 예측 토큰, 2단계(외형, SLat) = 실제 토큰
#  retr : 외형 = 예측 범주 안 train 정답 토큰 중 예측과 가장 가까운 것 하나 (slat_retrieve.py, fMRI만 씀)
#  gt   : 외형 = test 정답 토큰 (상한, 방법 아님)
#  1 검색  2 빠른 확인: 정성 14물체 × 2 Gaussian 렌더 → slat_quick.tar (SLAT_QUICK_DONE)
#  3 색 코드 실험(run_color.sh)이 끝날 때까지 기다림 (같은 GPU)  4 GLB 104 → 8지표  5 색 일치 · 정성 · 묶기 → slat.tar (SLAT_DONE)
#  비교 기준: fig = 이미 평가한 $SPP_ROOT/out/fig (예측 토큰을 두 단계 모두에 씀)
# 사용: nohup bash run_slatsplit.sh > $W/logs/prism_slat.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
PY=$W/venv/bin/python; I=$W/prism_renorm/fig/infer/sub-0001_keep; O=$W/prism_slat; VARS="retr gt"
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
sl() { [ $1 = gt ] && echo $TGT/dino || echo $O/slat_$1; }
cd $W/run; [ -f $I/components.npz ] || { echo "fig 추론 없음 $I (run_renorm.sh 결과)"; exit 1; }

step "1 검색 (slat_retrieve.py)"
$PY $PR/slat_retrieve.py --pred $I/tokens --comp $I/components.npz --tgt $TGT --data_root $DR --out $O/slat_retr | grep -E "SLATRET|없음|Error"
[ $(ls $O/slat_retr/*.npy 2>/dev/null | wc -l) -ge 104 ] || { echo "검색 결과가 104개 미만 $O/slat_retr"; exit 1; }

step "2 빠른 확인 (정성 14물체)"
for v in $VARS; do
  q=$O/quick/$v; [ $(cat $q/gscheck.jsonl 2>/dev/null | wc -l) -ge 14 ] && continue
  [ -d $q ] && mv $q $q.bad_$(date +%s)                                       # 끊긴 결과는 옆으로 치움 (rm 안 씀)
  trel gscheck --pred $I/tokens --slat_pred $(sl $v) --out $q --names $PR/scripts/qual14_uids.txt | grep -E "GSCHECK_DONE|Error|없음"
done
cd $O && tar cf slat_quick.tar $(for v in $VARS; do echo quick/$v/*.png quick/$v/gscheck.jsonl; done) slat_retr/slat_map.json && echo "PACK quick $(tar tf slat_quick.tar | grep -c png)개 (예상 28)"
step "SLAT_QUICK_DONE"

step "3 색 코드 실험 끝나기 기다림"
while pgrep -f run_color.sh > /dev/null; do sleep 60; done

step "4 GLB 104 → 8지표"
EV=""
for v in $VARS; do
  n=slat_$v; g=$O/glb/$n
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/tokens --slat_pred $(sl $v) --out $g | grep -E "GEN_DONE|Error"
  echo "GLB $n $(ls $g/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ -n "$EV" ] && wait $EV
  (cd $SPP_ROOT && bash run_spp.sh $n $g glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_$n.log 2>&1
   echo "EVAL $n 끝 $(date +%H:%M:%S) | $(tail -2 $W/logs/spp_$n.log | head -1)") &
  EV=$!
done
wait

step "5 색 일치 · 묶기"
P=$O/pack; mkdir -p $P
CA="fig=$SPP_ROOT/out/fig/render_a_L16000"; for v in $VARS; do CA="$CA slat_$v=$SPP_ROOT/out/slat_$v/render_a_L16000"; done
$PY $PR/scripts/color_agree.py --npz $PR/data/stim_color.npz --ids $DR/annotations/core_test_list.txt --train_ids $DR/annotations/core_train_list.txt \
  --json $P/color_agree.json $CA | tee $P/color_agree.txt
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for n in fig slat_retr slat_gt; do
  R=$SPP_ROOT/out/$n; mkdir -p $P/$n; cp $R/values.txt $P/$n/ 2>/dev/null || echo "지표 없음 $n"
  for u in $UIDS; do
    f=$(find $R/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
    [ -n "$f" ] && mkdir -p $P/$n/qual/$u && cp $f/{0,2,4}.png $P/$n/qual/$u/
  done
done
cp $O/slat_retr/slat_map.json $P/
cd $O && tar cf slat.tar pack && echo "PACK $(find pack -name '*.png' | wc -l)개 (예상 45) $(du -sh slat.tar | cut -f1)"
echo "=== 결과"; for n in fig slat_retr slat_gt; do echo "$n $(cat $P/$n/values.txt 2>/dev/null)"; done
step "SLAT_DONE"
