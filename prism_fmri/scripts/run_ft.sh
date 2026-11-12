#!/bin/bash
# ViT 미세조정 판정 (11-12, 5090): 범주 정보가 고정 ViT에서 빠짐(진단 A: 원 신호 0.41~0.44 → ViT 출력 0.26) → 피험자별 ViT 마지막 n블록 + norm을 같이 학습
#  기준 = run_reg.sh base와 같은 설정(fig 7명 bs 14, V = 범주 안 개체 형상, 시드 1, 검증 물체 8 × 13 뗌, 2만 스텝) + --enc_unfreeze n --enc_lr
#  고르는 기준 = [hold] 줄(검증 물체) 토큰 손실 · 범주 정확도. [val](= test)은 기록만
#  → $W/prism_ft/ft<n>, 로그 $W/logs/prism_ft_ft<n>.log (FT_DONE)
# 사용: GPU=1 N=2 nohup bash run_ft.sh > $W/logs/prism_ft.log 2>&1 < /dev/null & disown      (TEST=1: 60스텝 동작 확인, $W/prism_ft/test)
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
N=${N:-2}; LR=${ENC_LR:-1e-5}; STEPS=${STEPS:-20000}; SAVE=${SAVE:-5000,10000}; n=ft$N; X=""
[ "${TEST:-0}" = 1 ] && { n=test; STEPS=60; SAVE=""; X="--val_every 30 --limit 200"; }
PY=$W/venv/bin/python; ENC=$W/weights/MinD-3D-pp/fMRI-Shape; VT=$TGT/view_targets_shape.npz; O=$W/prism_ft
cd $W/run; mkdir -p $O
echo "=== 학습 시작 $n (마지막 $N블록, lr $LR) $(date +%H:%M:%S)"
$PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step $STEPS \
  --per_subject 1 --enc_dir $ENC --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 \
  --seed 1 --holdout 8 ${SAVE:+--save_steps $SAVE} --enc_unfreeze $N --enc_lr $LR $X --resume --out_dir $O/$n > $W/logs/prism_ft_$n.log 2>&1
echo "=== 학습 끝 $n $(date +%H:%M:%S) | $(grep -E '^\[hold\]' $W/logs/prism_ft_$n.log | tail -1 | cut -c1-200)"
echo "=== FT_DONE $(date +%H:%M:%S)"
