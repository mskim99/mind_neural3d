#!/bin/bash
# P-5 (11-28, Pro6000 GPU 1): Perceptive를 박사님 설계(Bank · Match to Bank · Peer Contrast → z_B → FiLM)로 바꿔 기존 Ours(ausvp) 설정 그대로 7명 학습
#  기존 Ours 설정 = prism_full/ausvp args 그대로 (UKB 고정 · in/feat norm · vit_lin 32 · branch · v_pool 2 · sv_enc 0 · shape head · bs 14 · 2만 · seed 1 · holdout 0)
#  더한 것: --bank 1 --bank_c 32 --bank_lam 0.9 (기준 객체 = train 1,300 전부, 학습 전 Bank를 처음 파라미터로 채움, 온도 없음)
#  MODE=check : 300 스텝 짧은 확인 → $W/prism_p5/check (Bank 채움 · loss · w 엔트로피 기록 확인)
#  MODE=temp  : P-5b (사용자 11-28 16:11) 학습되는 온도 τ (w = softmax(r / τ), 처음 0.07, 0.01 ~ 1) — 나머지는 full과 같음 → $W/prism_p5/ausvp_bank_temp
#  MODE=full  : 2만 스텝 → $W/prism_p5/ausvp_bank (WAIT_V10=1이면 logs/v10_s06.log에 V10_S06_DONE이 찍힐 때까지 기다렸다 시작)
# 사용: MODE=check nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_p5.sh > /home/oem/jhj/MinD-3D++/logs/p5_check.log 2>&1 < /dev/null & disown
GPU=1 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
GPU=1; export CUDA_VISIBLE_DEVICES=1; PY=$W/venv/bin/python; MODE=${MODE:-check}
case $MODE in check) O=$W/prism_p5/check; ST=300; VE=300; SV=300;; full) O=$W/prism_p5/ausvp_bank; ST=20000; VE=1000; SV=2000;; temp) O=$W/prism_p5/ausvp_bank_temp; ST=20000; VE=1000; SV=2000; XT="--bank_temp 1";; *) echo "MODE=check|full|temp"; exit 1;; esac
UKB=$W/weights/LEA_Visual_UKB/ukb_encoder.pt; VT=$TGT/view_targets_shape.npz
for f in $UKB $VT; do [ -f $f ] || { echo "없음 $f"; exit 1; }; done
if [ "$MODE" = full ] && [ "${WAIT_V10:-0}" = 1 ]; then
  echo "=== V-10 피험자 6번 끝 기다림 $(date +%H:%M:%S)"; until grep -q V10_S06_DONE $W/logs/v10_s06.log 2>/dev/null; do sleep 60; done
fi
wait_gpu_free p5_$MODE; mkdir -p $O; cd $W/run
echo "=== P-5 $MODE 시작 $(date +%H:%M:%S) → $O ($ST 스텝)"
$PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008 \
  --bs 14 --max_step $ST --val_every $VE --save_every $SV --save_steps 10000 --per_subject 1 --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT \
  --lam_vadv 0 --seed 1 --holdout 0 --enc_path $UKB --in_norm 1 --feat_norm 1 --vit_lin 32 --mem branch --v_pool 2 --sv_enc 0 \
  --bank 1 --bank_c 32 --bank_lam 0.9 ${XT:-} --resume --out_dir $O
echo "=== P5_${MODE^^}_DONE $(date +%H:%M:%S) $(ls $O/model_final.pt 2>/dev/null)"
