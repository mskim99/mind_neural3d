#!/bin/bash
# 토큰 prior 1단계 보고용 평가 (11-09, Pro6000 GPU 1): sub-0001 test 104, CFG는 run_prior1_tune.sh에서 고른 값 (prior 학습에서 뺀 train 물체 14 × sub-0002, test 물체 안 씀)
#  외형(SLat) = prior 생성 토큰, 형상(SS) = fig 예측 토큰 → GLB → 8지표 + 색 일치 + 복사 검사 → prior1.tar (PRIOR1_DONE)
#  비교: fig (예측 그대로) · slat_retr (검색, 채택) · slat_gt (정답 상한) = 이미 평가한 $SPP_ROOT/out
# 사용: CFG=2 nohup bash run_prior1_eval.sh > $W/logs/prism_prior1_eval.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
PY=$W/venv/bin/python; P1=$W/prism_prior1; I1=$P1/infer_fig/sub-0001_keep; E=$P1/eval
[ -n "$CFG" ] || { echo "CFG를 정해서 실행 (예: CFG=2 bash run_prior1_eval.sh)"; exit 1; }
n=prior_cfg$CFG
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; [ -f $P1/prior/prior_final.pt ] && [ -f $I1/components.npz ] || { echo "prior 또는 fig sub-0001 예측 없음"; exit 1; }

step "1 생성 (sub-0001 104, cfg $CFG)"
[ $(ls $E/gen_$n/*.npy 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/prior_token.py sample --ckpt $P1/prior/prior_final.pt --comp $I1/components.npz --out $E/gen_$n --cfg $CFG --tgt $TGT --data_root $DR | grep -E "생성|COPY|Error"

step "2 GLB → 8지표"
g=$E/glb/$n
[ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I1/tokens --slat_pred $E/gen_$n --out $g | grep -E "GEN_DONE|Error|없음"
echo "GLB $(ls $g/*/*.glb 2>/dev/null | wc -l)개 (예상 104)"
(cd $SPP_ROOT && bash run_spp.sh $n $g glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_$n.log 2>&1)
echo "EVAL $n 끝 | $(tail -2 $W/logs/spp_$n.log | head -1)"

step "3 색 일치 · 묶기"
P=$E/pack_$n; mkdir -p $P
$PY $PR/scripts/color_agree.py --npz $PR/data/stim_color.npz --ids $DR/annotations/core_test_list.txt --train_ids $DR/annotations/core_train_list.txt \
  --json $P/color_agree.json fig=$SPP_ROOT/out/fig/render_a_L16000 slat_retr=$SPP_ROOT/out/slat_retr/render_a_L16000 $n=$SPP_ROOT/out/$n/render_a_L16000 | tee $P/color_agree.txt
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
R=$SPP_ROOT/out/$n; cp $R/values.txt $P/ 2>/dev/null || echo "지표 없음 $n"
for u in $UIDS; do
  f=$(find $R/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
  [ -n "$f" ] && mkdir -p $P/qual/$u && cp $f/{0,2,4}.png $P/qual/$u/
done
cp $E/gen_$n/copy_check.json $P/
cd $E && tar cf prior1.tar pack_$n && echo "PACK $(find pack_$n -name '*.png' | wc -l)개 (예상 15)"
echo "=== 결과"; for x in fig slat_retr slat_gt $n; do echo "$x $(cat $SPP_ROOT/out/$x/values.txt 2>/dev/null)"; done
step "PRIOR1_DONE"
