#!/bin/bash
# 색 안 3 (11-06, Pro6000 GPU 1, 재학습 없음): TRELLIS 가이던스로 예측 토큰을 평균에서 밀어내면 색이 살아나는가
#  모델 prism_nop_feat (P 없음, feat), sub-0001 test 104개. 토큰은 이미 있음 (prism_infer_nop_feat/sub-0001_keep/tokens)
#  neg_cond: zero = 지금 (TRELLIS 기본) / mean = train 정답 DINO 토큰 평균. 가이던스 세기는 SLat(외형) 단계만 바꿈, -1 = TRELLIS 기본
#  1 평균 토큰  2 빠른 확인: 정성 14물체 × 설정 5개 Gaussian 렌더 + 색 (trellis_tools gscheck) → 묶기 (COLOR3_QUICK_DONE)
#  3 전체: 설정 4개 × GLB 104 → 8지표 (GLB와 평가를 겹침)  4 렌더 색 통계 · 묶기 (COLOR3_DONE)
# 사용: nohup bash run_color3.sh > $W/logs/prism_color3.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
PY=$W/venv/bin/python; T=$W/prism_infer_nop_feat/sub-0001_keep/tokens; O=$W/prism_color3; mkdir -p $O
step() { echo "=== $1 $(date +%H:%M:%S)"; }
SET_Q="zero:-1 mean:-1 zero:6 mean:6 mean:10"     # 빠른 확인 (첫 번째 = 지금 설정)
SET_F="mean:-1 zero:6 mean:6 mean:10"              # 전체 평가 (지금 설정은 spp_eval/out/nop_feat에 있음)
tag() { echo "${1%%:*}_s${1##*:}" | sed 's/s-1/def/'; }
gen() {  # $1 설정 $2 모드 $3 출력 [$4 --names 파일]
  n=${1%%:*}; c=${1##*:}
  (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && \
   python $PR/trellis_tools.py $2 --pred $T --out $3 --neg $n --neg_file $O/mean_tok.npy --slat_cfg $c ${4:+--names $4})
}
cd $W/run
[ -d $T ] || { echo "토큰 없음 $T"; exit 1; }

step "1 평균 토큰 (train 정답 DINO)"
[ -f $O/mean_tok.npy ] || $PY -c "
import numpy as np, os
ids = [l.strip().split('/')[1] for l in open('$DR/annotations/core_train_list.txt') if l.strip()]
fs = [f'$TGT/dino/{u}.npy' for u in ids if os.path.exists(f'$TGT/dino/{u}.npy')]; s = 0
for f in fs: s = s + np.load(f).astype(np.float64)
m = s / len(fs); np.save('$O/mean_tok.npy', m.astype(np.float32)); print('평균 토큰', len(fs), m.shape, 'norm', float(np.linalg.norm(m, axis=1).mean()))"

step "2 빠른 확인 (14물체 × ${SET_Q})"
for s in $SET_Q; do
  gen $s gscheck $O/quick/$(tag $s) $PR/scripts/qual14_uids.txt | grep -E "기본 sampler|neg |GSCHECK_DONE|Error"
  echo "  $(tag $s) 채도: $($PY -c "
import json; r=[json.loads(l) for l in open('$O/quick/$(tag $s)/gscheck.jsonl')]
print('Gaussian 밝기 %.1f 채도 %.1f | 텍스처 밝기 %.1f 채도 %.1f (n %d)' % tuple([sum(x[k][i] for x in r)/len(r) for k in ('gauss_mean_chroma','tex_mean_chroma') for i in (0,1)] + [len(r)]))")"
done
cd $O && tar cf color3_quick.tar quick/*/*.png quick/*/gscheck.jsonl && echo "PACK quick $(ls quick/*/*.png | wc -l)개 (예상 70)"; cd $W/run
step "COLOR3_QUICK_DONE"

step "3 전체 (GLB 104 → 8지표)"
EV=""
for s in $SET_F; do
  t=$(tag $s); gen $s gen $O/$t/glb | grep -E "GEN_DONE|Error"
  echo "GLB $t $(ls $O/$t/glb/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ -n "$EV" ] && wait $EV
  (cd $SPP_ROOT && bash run_spp.sh c3_$t $O/$t/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_c3_$t.log 2>&1
   echo "EVAL $t 끝 $(date +%H:%M:%S) | $(tail -2 $W/logs/spp_c3_$t.log | head -1)") &
  EV=$!
done
wait

step "4 렌더 색 · 묶기"
RUNS="zero_def=$SPP_ROOT/out/nop_feat/render_a_L16000"; for s in $SET_F; do RUNS="$RUNS $(tag $s)=$SPP_ROOT/out/c3_$(tag $s)/render_a_L16000"; done
$PY $PR/scripts/color_stats.py --tgt $TGT --ids $DR/annotations/core_test_list.txt $RUNS | tee $O/color_stats.txt
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
P=$O/pack; mkdir -p $P; cp $O/color_stats.txt $P/
for t in zero_def $(for s in $SET_F; do tag $s; done); do
  [ $t = zero_def ] && R=$SPP_ROOT/out/nop_feat || R=$SPP_ROOT/out/c3_$t
  mkdir -p $P/$t; cp $R/values.txt $P/$t/ 2>/dev/null || echo "지표 없음 $t"
  for u in $UIDS; do
    f=$(find $R/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
    [ -n "$f" ] && mkdir -p $P/$t/qual/$u && cp $f/{0,2,4}.png $P/$t/qual/$u/
  done
done
cd $O && tar cf color3.tar pack && echo "PACK $(find pack -name '*.png' | wc -l)개 (예상 75) $(du -sh color3.tar | cut -f1)"
echo "=== 결과"; for t in zero_def $(for s in $SET_F; do tag $s; done); do echo "$t $(cat $P/$t/values.txt 2>/dev/null)"; done
step "COLOR3_DONE"
