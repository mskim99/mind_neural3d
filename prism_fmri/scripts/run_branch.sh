#!/bin/bash
# 새 구조 1차 비교 (11-13, 사용자 승인) — prism.py 맨 위 "11-13 새 구조"
#  새 구조(--mem branch) = Semantic attention pooling 벡터 1개 + Variational head 토큰 6 + patch 토큰 6 × 256 + 갈래별 cross-attention
#  공통: fig 7명 bs 14, V = 범주 안 개체 형상, 시드 1, 검증 물체 8 × 13 뗌 (run_reg.sh · run_ukb.sh와 같음), 2만 스텝, 5천 스텝마다 저장
#  RUNS:  a    공개 피험자별 인코더 + 새 구조                         asv  a에서 SV Encoder 뺌 (--sv_enc 0)
#         ac   a + patch 토큰을 프레임 평균 + 프레임별 변화량으로 (--v_center 1)
#         au   UKB 인코더 4안 설정(고정 · 입력 z-score · 출력 표준화 · ViT 출력 선형 읽기 32) + 새 구조
#         aup · ausv · auc   au에서 patch 2×2 묶음 · SV Encoder 뺌 · 프레임 평균 + 변화량 (11-13, 5090에서: RUNS="aup ausv auc")
#         x · xp · xsv · xc   바탕 인코더 설정을 직접 줌: EX="--enc_path … --in_norm 1 …" (필수, 다른 바탕에서 같은 4종을 돌릴 때)
#  대조(이미 있음): 공개 인코더 이전 구조 = Ours vshape_g0, UKB 이전 구조 = 5090 prism_ukb/frozen_fn_vl
#  스텝 고르기: [hold] 줄(검증 물체) 토큰 손실 · 범주. 공개 인코더는 인코더가 검증 물체를 이미 봐서 [hold]가 낙관적 → a · asv · ac끼리 비교에만 씀
#  학습만 함 (추론 · GLB · 평가는 고른 것만 따로) → $W/prism_branch/<이름>, 로그 logs/prism_branch_<이름>.log (BRANCH_DONE)
# 사용: GPU=1 nohup bash run_branch.sh > $W/logs/prism_branch.log 2>&1 < /dev/null & disown
#       동작 확인(60스텝, 차례로): TEST=1 GPU=1 bash run_branch.sh        진행: SET=branch watch -n 1 bash status_reg.sh
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
RUNS=${RUNS:-"a asv ac ap au"}; STEPS=${STEPS:-20000}; SAVE=${SAVE:-5000,10000,15000}; HOLD=${HOLD:-8}; TEST=${TEST:-0}
PY=$W/venv/bin/python; PUB=$W/weights/MinD-3D-pp/fMRI-Shape; U=$W/weights/LEA_Visual_UKB/ukb_encoder.pt; VT=$TGT/view_targets_shape.npz; O=${OUT:-$W/prism_branch}; LP=prism_branch
EP="--enc_dir $PUB"; EU="--enc_path $U --in_norm 1 --feat_norm 1 --vit_lin 32"
flags() { case $1 in a) echo "$EP --mem branch";; asv) echo "$EP --mem branch --sv_enc 0";; ac) echo "$EP --mem branch --v_center 1";;
                     ap) echo "$EP --mem branch --v_pool 2";;     # a에서 patch 토큰을 이웃 2×2 평균 (6 × 64 = 384개)
                     au) echo "$EU --mem branch";;
                     aup) echo "$EU --mem branch --v_pool 2";; ausv) echo "$EU --mem branch --sv_enc 0";; auc) echo "$EU --mem branch --v_center 1";;
                     x) echo "$EX --mem branch";; xp) echo "$EX --mem branch --v_pool 2";; xsv) echo "$EX --mem branch --sv_enc 0";; xc) echo "$EX --mem branch --v_center 1";; *) echo BAD;; esac; }
EX=${EX:-}; for n in $RUNS; do case $n in x|xp|xsv|xc) [ -n "$EX" ] || { echo "x 계열은 EX(바탕 인코더 옵션)가 필요"; exit 1; };; esac; done
for n in $RUNS; do [ "$(flags $n)" = BAD ] && { echo "모르는 이름 $n"; exit 1; }; done
for f in $VT $U; do [ -f $f ] || { echo "없음: $f"; exit 1; }; done
X=""; [ $TEST = 1 ] && { O=$O/test; STEPS=60; SAVE=""; X="--val_every 30 --limit 200"; }
cd $W/run; mkdir -p $O
echo "디스크: $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"

one() {
  local n=$1 M=$O/$1 L=$W/logs/${LP}_$1.log; [ $TEST = 1 ] && L=$W/logs/${LP}_test_$1.log
  [ -f $M/model_final.pt ] && { echo "이미 있음 $n"; return; }
  echo "=== 학습 시작 $n ($(flags $n)) $(date +%H:%M:%S)"
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step $STEPS \
    --per_subject 1 --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 \
    --seed 1 --holdout $HOLD ${SAVE:+--save_steps $SAVE} $(flags $n) $X --resume --out_dir $M > $L 2>&1
  echo "=== 학습 끝 $n $(date +%H:%M:%S) | $(grep -E '^\[hold\]' $L | tail -1 | cut -c1-200)"
  [ $TEST = 1 ] && { grep -E 'Traceback|Error|out of memory' $L | tail -3; grep -oE 'step [0-9]+/[0-9]+.*' $L | tail -1 | cut -c1-160; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo; }
}
if [ $TEST = 1 ]; then for n in $RUNS; do one $n; done
else J=""; for n in $RUNS; do one $n & J="$J $!"; sleep 60; done; wait $J
  echo "=== 요약 (검증 물체 기준)"; SET=branch RUNS="$RUNS" bash $PR/scripts/status_reg.sh | sed -n '/^요약/,$p'; fi
echo "=== BRANCH_DONE $(date +%H:%M:%S)"
