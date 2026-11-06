#!/bin/bash
# 박사님 11-06 구조도 반영 모델 학습 · sub-0001 평가 (11-06, Pro6000 GPU 1)
#  구조 (train_prism.py 기본값): --arch fig (Temporal Stem · SV Encoder 구조도대로) --sem_clip front (CLIP Projection을 Down Stage 앞)
#    --var_heads sr (Scale + 3D 크기 비율 head) --var_mode feat. 학습 P 없음
#  학습 설정은 비교 기준선 prism_nop_feat과 같음 (7명 공동, 피험자별 ViT · S · V, bs 14, 4만 스텝) → sub-0001 지표를 spp_eval/out/nop_feat과 바로 비교
#  1 학습  2 sub-0001 토큰 → GLB → 8지표  3 정성(기준선과 같은 물체 5개 × 뷰 0·2·4) · 지표 묶기
# 사용: nohup bash run_fig.sh > $W/logs/prism_fig.log 2>&1 < /dev/null & disown
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
ENC=$W/weights/MinD-3D-pp/fMRI-Shape; PY=$W/venv/bin/python; S1=sub-0001
M=$W/prism_fig; I=$W/prism_infer_fig; CK=$M/model_final.pt
step() { echo "=== $1 $(date +%H:%M:%S)"; }
cd $W/run

step "1 학습 (arch fig · sem_clip front · var_heads sr, 7명 4만 스텝)"
if [ -f $CK ]; then echo "이미 있음 $CK"; else
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --init_ckpt $INIT --bs 14 --max_step 40000 \
    --per_subject 1 --enc_dir $ENC --arch fig --sem_clip front --var_heads sr --var_mode feat --out_dir $M > $W/logs/prism_fig_train.log 2>&1
fi
[ -f $CK ] || { echo "학습 실패: $W/logs/prism_fig_train.log"; tail -20 $W/logs/prism_fig_train.log; exit 1; }
echo "학습 끝 | $(grep -E '^\[val\]' $W/logs/prism_fig_train.log | tail -1)"

step "2 sub-0001 추론 → GLB → 8지표"
$PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I --subs $S1 --perceptive keep | grep -E "범주|DONE"
(cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py gen --pred $I/${S1}_keep/tokens --out $I/${S1}_keep/glb) | grep -E "GEN_DONE|Error"
echo "GLB $(ls $I/${S1}_keep/glb/*/*.glb 2>/dev/null | wc -l)개 (예상 104)"
(cd $SPP_ROOT && bash run_spp.sh fig $I/${S1}_keep/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_fig.log 2>&1)
echo "EVAL 끝 | $(tail -2 $W/logs/spp_fig.log | head -1)"

step "3 묶기 (기준선 nop_feat과 나란히)"
O=$W/prism_attr/fig; mkdir -p $O
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for m in nop_feat fig; do
  R=$SPP_ROOT/out/$m; mkdir -p $O/$m; cp $R/values.txt $O/$m/ 2>/dev/null || echo "지표 없음 $m"
  for u in $UIDS; do
    f=$(find $R/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
    [ -n "$f" ] && mkdir -p $O/$m/qual/$u && cp $f/{0,2,4}.png $O/$m/qual/$u/
  done
done
cp $M/train_log.csv $W/logs/prism_fig_train.log $O/fig/ 2>/dev/null
cd $W/prism_attr && tar cf fig.tar fig && echo "PACK $(find fig -name '*.png' | wc -l)개 (예상 30) $(du -sh fig.tar | cut -f1)"
echo "=== 결과"; for m in nop_feat fig; do echo "$m $(cat $O/$m/values.txt 2>/dev/null)"; done
step "FIG_DONE"
