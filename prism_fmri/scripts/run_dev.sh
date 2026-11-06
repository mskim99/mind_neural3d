#!/bin/bash
# 편차 손실 실험 직렬 실행 (11-05 정성 진단 후속, Pro6000 GPU 1). 사용: bash run_dev.sh   로그 logs/prism_dev.log
#  1 기준선(prism_shape_ps7 final) test 재평가 (dev_cos 포함)  2 편차 손실 + InfoNCE로 7명 학습 1.5만 스텝 (val 500마다, best 저장)
#  3 두 모델 test 토큰 추론  4 피험자별 토큰 진단 (기준선·새 모델)  5 Subject 1 qual13 미리보기 (두 모델)  6 새 모델 best로 7명 GLB 생성
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
ENC=$W/weights/MinD-3D-pp/fMRI-Shape; D=$W/prism_diag; DV=$W/prism_diag_dev
BASE=$W/prism_shape_ps7/model_final.pt; NEW=$W/prism_shape_dev7
COMMON="--repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --init_ckpt $INIT --bs 14 --per_subject 1 --enc_dir $ENC"
PY=$W/venv/bin/python
step() { echo "=== $1 $(date +%H:%M:%S)"; }
set -e; mkdir -p $DV
cd $W/run
step "1 기준선 재평가"; $PY $PR/train_prism.py $COMMON --out_dir $W/prism_eval_ps7 --eval_ckpt $BASE
step "2 편차 손실 학습"; $PY $PR/train_prism.py $COMMON --out_dir $NEW --max_step 15000 --val_every 500 --save_every 2000 --dev_loss 1 --lam_nce 0.2
step "3 토큰 추론"
$PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $BASE --out $W/prism_infer_ps7 --perceptive keep
$PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $NEW/model_best.pt --out $W/prism_infer_dev7 --perceptive keep
step "4 토큰 진단"
for m in ps7 dev7; do for s in ${SUBS//,/ }; do
  U=/dev/null; [ $s = sub-0001 ] && U=$D/scripts/qual13_uids.txt      # 렌더용 변형 토큰은 피험자 1만 (나머지는 통계만)
  $PY $D/scripts/diag_tokens.py --pred $W/prism_infer_$m/${s}_keep/tokens --gt $TGT/dino --data_root $DR --uids $U --out $DV/${m}_$s | grep -E "SUMMARY|rescale"
done; done
step "5 qual13 미리보기"
(cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && for m in ps7 dev7; do python $D/scripts/diag_gen.py --diag $DV/${m}_sub-0001 --variants pred; done)
cd $DV && tar cf dev_diag.tar */token_stats.json */png_pred $NEW/train_log.csv $W/logs/prism_dev.log 2>/dev/null || true
echo "DEV_DIAG_DONE $(ls $DV/*/png_pred/*.png | wc -l)개"
step "6 새 모델 GLB 생성"
for s in ${SUBS//,/ }; do
  (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py gen --pred $W/prism_infer_dev7/${s}_keep/tokens --out $W/prism_infer_dev7/${s}_keep/glb)
  echo "GLB $s $(ls $W/prism_infer_dev7/${s}_keep/glb/*.glb | wc -l)개"
done
step "ALL_DONE"
