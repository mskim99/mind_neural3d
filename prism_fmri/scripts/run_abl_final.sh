#!/bin/bash
# 최종 Ours(ausvp, 1,300개 · 7명 · 2만) Ablation (11-15, 사용자) — 11-11 덱 Ablation 장과 같은 정의 ("뺀다" = 그 갈래를 가르치는 손실을 끔, 구조는 그대로)
#  noS = --lam_proto 0 / noV = --lam_var 0 / noP = --lam_subj 0 --film 0 / onlyS = noV + noP / onlyV = noS + noP / onlyP = noS + noV / noSVP = 셋 다
#  full = 이미 있음 (prism_full/ausvp, Subject 1 = yaw_fullausvp_s01)
#  학습: 최종과 같은 설정(7명 · 1,300개 · 검증 없음 · 2만 스텝) → 평가: Subject 1만 8지표 (run_sub_eval.sh NS=01, TAG=abl_<이름>)
#  GPU마다 NAMES를 차례로: 두 개씩 동시 학습 → 끝난 것부터 평가는 그 GPU에서 하나씩 (GLB 생성과 평가 렌더가 겹치지 않게)
# 사용: GPU=0 NAMES="noS noV noP onlyS" nohup bash run_abl_final.sh > $W/logs/abl_final_g0.log 2>&1 < /dev/null & disown
#       WAIT_DIR=<GLB 폴더> WAIT_N=<개수>: 같은 GPU의 다른 GLB 생성이 끝날 때까지 기다렸다 시작 (GLB 21 GB + 학습 2개 12 GB면 32 GB를 넘음)
W=/home/oem/jhj/MinD-3D++; S=$W/mind_neural3d/prism_fmri/scripts; O=$W/prism_full_abl
EU="--enc_path $W/weights/LEA_Visual_UKB/ukb_encoder.pt --in_norm 1 --feat_norm 1 --vit_lin 32 --mem branch --v_pool 2 --sv_enc 0"
flags() { case $1 in noS) echo "--lam_proto 0";; noV) echo "--lam_var 0";; noP) echo "--lam_subj 0 --film 0";;
                     onlyS) echo "--lam_var 0 --lam_subj 0 --film 0";; onlyV) echo "--lam_proto 0 --lam_subj 0 --film 0";; onlyP) echo "--lam_proto 0 --lam_var 0";;
                     noSVP) echo "--lam_proto 0 --lam_var 0 --lam_subj 0 --film 0";; *) echo BAD;; esac; }
for n in ${NAMES:?}; do [ "$(flags $n)" = BAD ] && { echo "모르는 이름 $n"; exit 1; }; done
[ -n "$WAIT_DIR" ] && { echo "=== 대기 $WAIT_DIR ${WAIT_N}개 $(date +%H:%M:%S)"; until [ $(ls $WAIT_DIR/*/*.glb 2>/dev/null | wc -l) -ge ${WAIT_N:-104} ]; do sleep 60; done; sleep 120; }
source $S/common.sh >/dev/null || exit 1; export CUDA_VISIBLE_DEVICES=$GPU
PY=$W/venv/bin/python; VT=$TGT/view_targets_shape.npz; mkdir -p $O; cd $W/run
train() { local n=$1; [ -f $O/$n/model_final.pt ] && return
  echo "=== 학습 $n ($(flags $n)) $(date +%H:%M:%S)"
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008 --bs 14 --max_step 20000 \
    --per_subject 1 --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 --seed 1 --holdout 0 $EU $(flags $n) --resume --out_dir $O/$n > $W/logs/abl_final_$n.log 2>&1
  echo "=== 학습 끝 $n $(date +%H:%M:%S) $(ls $O/$n/model_final.pt 2>/dev/null)"; }
set -- $NAMES
while [ $# -gt 0 ]; do                                   # 두 개씩 동시 학습
  a=$1; b=$2; train $a & PA=$!; [ -n "$b" ] && { sleep 60; train $b & PB=$!; }
  wait $PA ${PB:-}; PB=""; shift; [ -n "$b" ] && shift
done
for n in $NAMES; do                                      # Subject 1 평가 (하나씩)
  [ -f $O/$n/model_final.pt ] || { echo "학습 실패 $n"; continue; }
  echo "=== 평가 $n $(date +%H:%M:%S)"
  GPU=$GPU NS="01" CK=$O/$n/model_final.pt CK_ROOT=$O/eval_$n TAG=abl_$n bash $S/run_sub_eval.sh > $W/logs/abl_final_eval_$n.log 2>&1
  echo "CV|$n $(grep -E '^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' /home/oem/jhj/fair_eval/fair_eval/out/yaw_abl_${n}_s01/values.txt 2>/dev/null | awk '{printf "%s ", $2}') $(date +%H:%M:%S)"
done
echo "=== ABL_FINAL_DONE $(date +%H:%M:%S)"
