#!/bin/bash
# Variational 입력 비교 나머지 6명 평가 (11-06, Pro6000 GPU 1). 학습은 run_var_cmp.sh에서 끝남 (prism_nop_{feat,value}, 7명 공동)
#  두 모델 병렬. 모델마다: 6명 토큰 → 자세 일치도 → 피험자별 GLB · 8지표 (GLB와 평가를 겹침, 평가는 모델당 한 번에 하나)
#  sub-0001은 이미 있음 (spp_eval/out/nop_<m>) → 묶을 때 함께 넣음
# 사용: nohup bash run_var_cmp7.sh > $W/logs/prism_varcmp7.log 2>&1 < /dev/null & disown
SUBS6="sub-0002 sub-0003 sub-0005 sub-0006 sub-0007 sub-0008"
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
PY=$W/venv/bin/python; CSV=${SUBS6// /,}
step() { echo "=== $1 $(date +%H:%M:%S)"; }
cd $W/run

chain() {
  m=$1; I=$W/prism_infer_nop_$m; CK=$W/prism_nop_$m/model_final.pt; EV=""
  $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I --subs $CSV --perceptive keep | grep -E "범주|DONE" | sed "s/^/[$m] /"
  $PY $PR/eval_pose.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --ckpt $CK --out $I --subs $CSV | grep -E "POSE " | sed "s/^/[$m] /"
  for s in $SUBS6; do
    (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py gen --pred $I/${s}_keep/tokens --out $I/${s}_keep/glb) | grep -E "Error"
    echo "[$m] GLB $s $(ls $I/${s}_keep/glb/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
    [ -n "$EV" ] && wait $EV
    (cd $SPP_ROOT && bash run_spp.sh nop_${m}_$s $I/${s}_keep/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_nop_${m}_$s.log 2>&1
     echo "[$m] EVAL $s 끝 $(date +%H:%M:%S) | $(tail -2 $W/logs/spp_nop_${m}_$s.log | head -1)") &
    EV=$!
  done
  wait
}
step "1 6명 추론 · 자세 · GLB · 8지표 (두 모델 병렬)"
chain feat & chain value & wait

step "2 결과 묶기"
O=$W/prism_attr/varcmp7; mkdir -p $O
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for m in feat value; do
  for s in sub-0001 $SUBS6; do
    [ $s = sub-0001 ] && R=$SPP_ROOT/out/nop_$m || R=$SPP_ROOT/out/nop_${m}_$s
    mkdir -p $O/$m/$s; cp $R/values.txt $O/$m/$s/ 2>/dev/null || echo "지표 없음 $m $s"
    cp $W/prism_infer_nop_$m/pose_$s.* $O/$m/$s/ 2>/dev/null
    for u in $UIDS; do
      f=$(find $R/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
      [ -n "$f" ] && mkdir -p $O/$m/$s/qual/$u && cp $f/{0,2,4}.png $O/$m/$s/qual/$u/
    done
  done
done
cd $W/prism_attr && tar cf varcmp7.tar varcmp7 && echo "PACK $(find varcmp7 -name '*.png' | wc -l)개 (예상 210) $(du -sh varcmp7.tar | cut -f1)"
echo "=== 결과"; for m in feat value; do for s in sub-0001 $SUBS6; do echo "$m $s $(cat $O/$m/$s/values.txt 2>/dev/null)"; done; done
step "VARCMP7_DONE"
