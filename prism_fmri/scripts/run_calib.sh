#!/bin/bash
# 11-17 추가 논문 Perceptive 개선안 1단계: 피험자 · 영역별 입력 보정만 학습 (train_calib.py) → $OUT/calib.pt
#  고정 UKB 인코더 앞 보정, 같은 물체 · 다른 피험자 ViT 특징을 가깝게. 끝에 [eval] 보정 전(0스텝) vs 후 — hold gap이 오르고 top1이 안 떨어지면 "정렬됨"
#  그 뒤 2단계: RUNS="base_sharedsv base_sharedsv_calib base_vres_sharedsv_calib" run_branch.sh — _calib 이름은 CAL=$OUT/calib.pt (기본 $W/prism_calib/s16/calib.pt)
# 사용: GPU=1 SUBS=sub-0001,sub-0006 nohup bash run_calib.sh > $W/logs/prism_calib_s16.log 2>&1 < /dev/null & disown
#  공용 서버 배려: 전체를 nice로 띄우고 감시는 polite_watch.sh
#  진행: watch -n 1 "grep -aE '^(step|\[eval\]|CALIB)' $W/logs/prism_calib_s16.log | tail -5"
SUBS=${SUBS:-sub-0001,sub-0006}
source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; U=$W/weights/LEA_Visual_UKB/ukb_encoder.pt; OUT=${OUT:-$W/prism_calib/s16}; STEPS=${STEPS:-3000}; TEST=${TEST:-0}
X=""; [ $TEST = 1 ] && { OUT=$OUT/test; X="--steps 40 --eval_every 20 --print_every 10 --limit 40"; }
mkdir -p $OUT; cd $W/run
echo "=== 1단계 시작 $(date +%H:%M:%S) | 피험자 $SUBS | GPU $CUDA_VISIBLE_DEVICES | 디스크: $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
$PY $PR/train_calib.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --enc_path $U --in_norm 1 --holdout ${HOLD:-8} \
  --steps $STEPS --polite ${POLITE:-65} --out_dir $OUT $X ${EXTRA:-}
echo "=== 1단계 끝 $(date +%H:%M:%S)"
