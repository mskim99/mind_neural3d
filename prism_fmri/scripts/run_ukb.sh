#!/bin/bash
# UKB 사전학습 인코더로 PRISM 학습 (11-12, 5090, 사용자 승인 "고정으로 시작")
#  근거: 공개 피험자별 인코더는 fMRI-Shape train 1,300개를 학습하며 외움(검증 물체 0.88 / test 0.27). fMRI-Shape를 본 적 없는 UKB 사전학습 인코더는
#        고정 · 입력 프레임별 z-score에서 검증 물체 0.29 / test 0.39 (원 신호 0.30 / 0.44) — `.보고 자료/261112_9차보고/판정실험/성능개선1_5090.md`
#  설정 = run_reg.sh base와 같음(fig 7명 bs 14, V = 범주 안 개체 형상, 시드 1, 검증 물체 8 × 13 뗌, 2만 스텝) + 인코더만 바꿈
#    NAME=frozen : UKB 인코더(피험자 공통) 고정 + --in_norm 1        NAME=ft<n> : + 마지막 n블록 풀기 (N=n)
#  이 인코더에서는 검증 물체가 "아무도 안 본 물체" → [hold]로 고를 수 있음
#  → $W/prism_ukb/<이름>, 로그 $W/logs/prism_ukb_<이름>.log (UKB_DONE)
# 사용: GPU=0 nohup bash run_ukb.sh > $W/logs/prism_ukb.log 2>&1 < /dev/null & disown      (TEST=1: 60스텝 동작 확인 → $W/prism_ukb/test)
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
N=${N:-0}; STEPS=${STEPS:-20000}; SAVE=${SAVE:-5000,10000}; n=frozen; X=""; [ $N -gt 0 ] && { n=ft$N; X="--enc_unfreeze $N --enc_lr ${ENC_LR:-1e-5}"; }
[ "${TEST:-0}" = 1 ] && { n=test; STEPS=60; SAVE=""; X="$X --val_every 30 --limit 200"; }
PY=$W/venv/bin/python; UKB=$W/weights/LEA_Visual_UKB/ukb_encoder.pt; VT=$TGT/view_targets_shape.npz; O=$W/prism_ukb
cd $W/run; mkdir -p $O
echo "=== 학습 시작 $n $(date +%H:%M:%S)"
$PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step $STEPS \
  --per_subject 1 --enc_path $UKB --in_norm 1 --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 \
  --seed 1 --holdout 8 ${SAVE:+--save_steps $SAVE} $X --resume --out_dir $O/$n > $W/logs/prism_ukb_$n.log 2>&1
echo "=== 학습 끝 $n $(date +%H:%M:%S) | $(grep -E '^\[hold\]' $W/logs/prism_ukb_$n.log | tail -1 | cut -c1-200)"
echo "=== UKB_DONE $(date +%H:%M:%S)"
