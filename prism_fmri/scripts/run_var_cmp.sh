#!/bin/bash
# Variational 입력 비교 (11-05, Pro6000 GPU 1). 두 모델 모두 학습 P 없음, 7명 학습 설정은 prism_shape_ps7과 같음 (bs 14, 4만 스텝)
#  11-06 기본값이 바뀌어 당시 구조(--arch res --sem_clip back --var_heads csr)를 명시해 고정함 → prism_nop_feat · value 재학습용
#  feat  = 기준선: head의 LayerNorm 특징 → Aggregate      value = head가 낸 center · scale · 회전 값 → MLP 임베딩 → Aggregate
#  1 두 학습 병렬  2 모델마다 병렬로: sub-0001 토큰 → 자세 일치도(eval_pose.py) → GLB → 8지표(spp_eval)  3 결과 묶기
# 사용: nohup bash run_var_cmp.sh > $W/logs/prism_varcmp.log 2>&1 < /dev/null & disown
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
ENC=$W/weights/MinD-3D-pp/fMRI-Shape; PY=$W/venv/bin/python; S1=sub-0001
COMMON="--repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --init_ckpt $INIT --bs 14 --max_step 40000 --per_subject 1 --enc_dir $ENC --arch res --sem_clip back --var_heads csr"
step() { echo "=== $1 $(date +%H:%M:%S)"; }
cd $W/run

step "1 학습 시작 (feat · value 병렬)"
for m in feat value; do
  $PY $PR/train_prism.py $COMMON --var_mode $m --out_dir $W/prism_nop_$m > $W/logs/prism_nop_$m.log 2>&1 &
done
wait
for m in feat value; do echo "학습 $m: $(ls $W/prism_nop_$m/model_final.pt 2>/dev/null || echo 없음) | $(grep -E '^\[val\]|val ' $W/logs/prism_nop_$m.log | tail -1)"; done

step "2 sub-0001 추론 · 자세 · GLB · 8지표 (두 모델 병렬)"
chain() {
  m=$1; I=$W/prism_infer_nop_$m; CK=$W/prism_nop_$m/model_final.pt
  $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I --subs $S1 --perceptive keep | grep -E "범주|DONE"
  $PY $PR/eval_pose.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --ckpt $CK --out $I --subs $S1 | grep -E "POSE"
  (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py gen --pred $I/${S1}_keep/tokens --out $I/${S1}_keep/glb) | grep -E "GEN_DONE|Error"
  echo "GLB $m $(ls $I/${S1}_keep/glb/*/*.glb 2>/dev/null | wc -l)개"
  (cd $SPP_ROOT && bash run_spp.sh nop_$m $I/${S1}_keep/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_nop_$m.log 2>&1)
  echo "=== EVAL $m 끝 $(date +%H:%M:%S) | $(tail -2 $W/logs/spp_nop_$m.log | head -1)"
}
chain feat & chain value & wait

step "3 결과 묶기"
O=$W/prism_attr/varcmp; mkdir -p $O
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for m in feat value; do
  mkdir -p $O/$m; cp $SPP_ROOT/out/nop_$m/values.txt $O/$m/ 2>/dev/null; cp $W/prism_infer_nop_$m/pose_$S1.* $O/$m/ 2>/dev/null
  cp $W/prism_nop_$m/train_log.csv $O/$m/ 2>/dev/null; cp $W/logs/prism_nop_$m.log $O/$m/ 2>/dev/null
  for u in $UIDS; do
    f=$(find $SPP_ROOT/out/nop_$m/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
    [ -n "$f" ] && mkdir -p $O/$m/qual/$u && cp $f/{0,2,4}.png $O/$m/qual/$u/
  done
done
cd $W/prism_attr && tar cf varcmp.tar varcmp && echo "PACK $(find varcmp -name '*.png' | wc -l)개 (예상 30) $(du -sh varcmp.tar | cut -f1)"
echo "=== 결과"; for m in feat value; do echo "$m $(cat $O/$m/values.txt 2>/dev/null)"; done
step "VARCMP_DONE"
