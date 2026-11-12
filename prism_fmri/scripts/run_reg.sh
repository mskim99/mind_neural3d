#!/bin/bash
# 과적합 대책 1단계 (11-12, 사용자 승인) — 계획 `.보고 자료/261112_9차보고/PRISM_개선/성능개선_가설·계획.md`
#  근거: Ours(vshape_g0)의 검증 토큰 손실이 5천 스텝 0.266 → 4만 스텝 0.315 (학습 0.089), 범주 학습 0.99 vs 검증 0.29
#  기준 = vshape_g0 설정(fig 7명 bs 14, V = 범주 안 개체 형상, GRL 없음, 시드 1) + 공통: 2만 스텝 · 검증 물체(train 범주마다 목록 끝 8개 = 104개)를 학습에서 뺌
#  RUNS:  base  기준 그대로                      do    dropout 0.1                  dowd  dropout 0.1 + weight decay 0.05
#         aug   fMRI 증강(잡음 0.1 · 패치 가림 15%)   augdo 증강 + dropout 0.1           lin   Semantic 안 원 신호 선형 경로 (+ dowd와 같은 정규화 → dowd와 짝 비교)
#  고르는 기준 = [hold] 줄(검증 물체)의 토큰 손실 · 피험자별 범주 정확도. [val](= test)은 기록만 하고 고르는 데 쓰지 않음
#  학습만 함 (추론 · GLB · 평가는 이긴 설정만 따로). 체크포인트 model_step{5000,10000}.pt + model_final.pt
#  → $W/prism_reg/<이름>, 끝에 요약 표 (REG_DONE)
#  UKB=1 (11-12 저녁): 인코더만 UKB 사전학습(피험자 공통, 고정, 입력 프레임별 z-score)으로 바꿈 — 공개 피험자별 인코더는 train 1,300개를 외워 [hold]를 못 씀,
#    UKB 인코더에서는 검증 물체가 아무도 안 본 물체 → [hold]로 고를 수 있음. 기준(base) = 5090 `prism_ukb/frozen`(같은 설정)이라 기본 RUNS에서 뺌
#    → $W/prism_reg_ukb/<이름>, 로그 logs/prism_regukb_<이름>.log
# 사용: GPU=1 nohup bash run_reg.sh > $W/logs/prism_reg.log 2>&1 < /dev/null & disown
#       UKB=1 GPU=1 nohup bash run_reg.sh > $W/logs/prism_regukb.log 2>&1 < /dev/null & disown
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
UKB=${UKB:-0}; DEF="base do dowd aug augdo lin"; [ $UKB = 1 ] && DEF="do dowd aug augdo lin"
RUNS=${RUNS:-$DEF}; STEPS=${STEPS:-20000}; SAVE=${SAVE:-5000,10000}; HOLD=${HOLD:-8}
PY=$W/venv/bin/python; ENC=${ENC:-$W/weights/MinD-3D-pp/fMRI-Shape}; VT=$TGT/view_targets_shape.npz; O=${OUT:-$W/prism_reg}; LP=prism_reg; EA="--enc_dir $ENC"
[ $UKB = 1 ] && { U=$W/weights/LEA_Visual_UKB/ukb_encoder.pt; [ -f $U ] || { echo "없음: $U"; exit 1; }; O=${OUT:-$W/prism_reg_ukb}; LP=prism_regukb; EA="--enc_path $U --in_norm 1"; }
:   # 5090 이른 체크포인트 판정: RUNS=base STEPS=40000 SAVE=2000,5000,10000,20000 HOLD=0 OUT=$W/prism_ckpt
flags() { case $1 in base) echo "";; do) echo "--dropout 0.1";; dowd) echo "--dropout 0.1 --wd 0.05";;
                     aug) echo "--aug_noise 0.1 --aug_mask 0.15";; augdo) echo "--aug_noise 0.1 --aug_mask 0.15 --dropout 0.1";;
                     lin) echo "--lin_path 1 --dropout 0.1 --wd 0.05";; *) echo BAD;; esac; }
for n in $RUNS; do [ "$(flags $n)" = BAD ] && { echo "모르는 이름 $n"; exit 1; }; done
[ -f $VT ] || { echo "없음: $VT"; exit 1; }
cd $W/run; mkdir -p $O
echo "디스크: $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"

one() {
  local n=$1 M=$O/$1
  [ -f $M/model_final.pt ] && { echo "이미 있음 $n"; return; }
  echo "=== 학습 시작 $n ($(flags $n)) $(date +%H:%M:%S)"
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step $STEPS \
    --per_subject 1 $EA --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 \
    --seed 1 --holdout $HOLD --save_steps $SAVE $(flags $n) --resume --out_dir $M > $W/logs/${LP}_$n.log 2>&1
  echo "=== 학습 끝 $n $(date +%H:%M:%S) | $(grep -E '^\[hold\]' $W/logs/${LP}_$n.log | tail -1 | cut -c1-200)"
}
J=""; for n in $RUNS; do one $n & J="$J $!"; sleep 60; done
wait $J
echo "=== 요약 (검증 물체 기준)"
UKB=$UKB RUNS="$RUNS" bash $PR/scripts/status_reg.sh | sed -n '/^요약/,$p'
echo "=== REG_DONE $(date +%H:%M:%S)"
