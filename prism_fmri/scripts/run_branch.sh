#!/bin/bash
# 새 구조 1차 비교 (11-13, 사용자 승인) — prism.py 맨 위 "11-13 새 구조"
#  새 구조(--mem branch) = Semantic attention pooling 벡터 1개 + Variational head 토큰 6 + patch 토큰 6 × 256 + 갈래별 cross-attention
#  공통: fig 7명 bs 14, V = 범주 안 개체 형상, 시드 1, 검증 물체 8 × 13 뗌 (run_reg.sh · run_ukb.sh와 같음), 2만 스텝, 5천 스텝마다 저장
#  RUNS:  a    공개 피험자별 인코더 + 새 구조                         asv  a에서 SV Encoder 뺌 (--sv_enc 0)
#         ac   a + patch 토큰을 프레임 평균 + 프레임별 변화량으로 (--v_center 1)
#         au   UKB 인코더 4안 설정(고정 · 입력 z-score · 출력 표준화 · ViT 출력 선형 읽기 32) + 새 구조
#         ausvp   au에서 patch 2×2 묶음 + SV Encoder 뺌 (11-13, 5090: SV Encoder 유무 × 묶음 유무 2×2 비교의 네 번째 칸)
#         aup · ausv · auc   au에서 patch 2×2 묶음 · SV Encoder 뺌 · 프레임 평균 + 변화량 (11-13, 5090에서: RUNS="aup ausv auc")
#         x · xp · xsv · xc   바탕 인코더 설정을 직접 줌: EX="--enc_path … --in_norm 1 …" (필수, 다른 바탕에서 같은 4종을 돌릴 때)
#  대조(이미 있음): 공개 인코더 이전 구조 = Ours vshape_g0, UKB 이전 구조 = 5090 prism_ukb/frozen_fn_vl
#  스텝 고르기: [hold] 줄(검증 물체) 토큰 손실 · 범주. 공개 인코더는 인코더가 검증 물체를 이미 봐서 [hold]가 낙관적 → a · asv · ac끼리 비교에만 씀
#  학습만 함 (추론 · GLB · 평가는 고른 것만 따로) → $W/prism_branch/<이름>, 로그 logs/prism_branch_<이름>.log (BRANCH_DONE)
# 사용: GPU=1 nohup bash run_branch.sh > $W/logs/prism_branch.log 2>&1 < /dev/null & disown
#       동작 확인(60스텝, 차례로): TEST=1 GPU=1 bash run_branch.sh        진행: SET=branch watch -n 1 bash status_reg.sh
SUBS=${SUBS:-sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008}   # 11-15: 2명 학습은 SUBS=sub-0001,sub-0006
source $(dirname $0)/common.sh || exit 1
RUNS=${RUNS:-"a asv ac ap au"}; STEPS=${STEPS:-20000}; SAVE=${SAVE:-5000,10000,15000}; HOLD=${HOLD:-8}; TEST=${TEST:-0}
PY=$W/venv/bin/python; PUB=$W/weights/MinD-3D-pp/fMRI-Shape; U=$W/weights/LEA_Visual_UKB/ukb_encoder.pt; VT=$TGT/view_targets_shape.npz; O=${OUT:-$W/prism_branch}; LP=${LP:-prism_branch}   # OUT · LP: 다른 스텝 수로 처음부터 다시 돌릴 때 폴더 · 로그 이름을 따로 (예: OUT=$W/prism_branch40 LP=prism_branch40 STEPS=40000)
EP="--enc_dir $PUB"; EU="--enc_path $U --in_norm 1 --feat_norm 1 --vit_lin 32"
flags() { case $1 in g0) echo "$EP";;   # 11-15: 기존 Ours 설정 (공개 인코더 + 이전 구조 cat), 2명 학습 대조용
                     a) echo "$EP --mem branch";; asv) echo "$EP --mem branch --sv_enc 0";; ac) echo "$EP --mem branch --v_center 1";;
                     ap) echo "$EP --mem branch --v_pool 2";;     # a에서 patch 토큰을 이웃 2×2 평균 (6 × 64 = 384개)
                     au) echo "$EU --mem branch";;
                     aup) echo "$EU --mem branch --v_pool 2";; ausvp) echo "$EU --mem branch --v_pool 2 --sv_enc 0";;   # 11-13: patch 2×2 묶음 + SV Encoder 뺌 (aup · ausv를 합친 것, SV Encoder × 묶음 2×2 비교용)
                     ausvpl) echo "$EU --mem branch --v_pool 2 --sv_enc 0 --sem_loc ${SL:-4}";;   # 11-15 A: ausvp + Semantic 위치 토큰 g×g (SL, 기본 4 → 16개)
                     ausvpn) echo "$EU --mem branch --v_pool 2 --sv_enc 0 --lam_nce ${LN:-0.1} --nce_temp ${NT:-0.02}";;   # 11-15 C: ausvp + 출력 토큰 대비 손실
                     ausvpln) echo "$EU --mem branch --v_pool 2 --sv_enc 0 --sem_loc ${SL:-4} --lam_nce ${LN:-0.1} --nce_temp ${NT:-0.02}";;   # 11-15 A + C
                     aupvr) echo "$EU --mem branch --v_pool 2 --v_res_basis ${VRB:-$TGT/vres_basis_k32.npz}";;   # 11-17 추가 논문 Variational 임시 실험 (= base_vres, 이미 이 이름으로 학습 중이라 남김) — 먼저 prep_vres_basis.py
                     # 11-17 추가 논문 이름 규칙 (사용자): 지금 기준 = base (= aup: UKB 인코더 + 새 구조 + patch 2×2 묶음, SV Encoder 있음), 더한 것은 이름에 다 붙임
                     base) echo "$EU --mem branch --v_pool 2";;
                     base_vres) echo "$EU --mem branch --v_pool 2 --v_res_basis ${VRB:-$TGT/vres_basis_k32.npz}";;   # + Variational 잔차 (범주 평균 토큰 + PCA 32, = aupvr)
                     base_sharedsv) echo "$EU --mem branch --v_pool 2 --shared_sv 1 --film 0 --lam_subj 0";;   # + S · V 피험자 공통 (이전 Perceptive FiLM · 피험자 분류 끔), 보정 없음 — 대조
                     base_sharedsv_calib) echo "$EU --mem branch --v_pool 2 --shared_sv 1 --film 0 --lam_subj 0 --calib 1 --calib_init ${CAL:-$W/prism_calib/s16/calib.pt}";;   # + 1단계 영역별 보정(고정) = Perceptive 개선안
                     base_sharedsv_calib_bound_keep) echo "$EU --mem branch --v_pool 2 --shared_sv 1 --film 0 --lam_subj 0 --calib 1 --calib_init ${CAL:?CAL 필요 (크기 묶기 + 보존 항 1단계 calib.pt)}";;   # 11-18 보정 다시: 크기 묶기(bound · renorm) + 보존 항(keep), 설정은 calib.pt에 들어 있음
                     ausvp_sharedsv) echo "$EU --mem branch --v_pool 2 --sv_enc 0 --shared_sv 1 --film 0 --lam_subj 0";;   # 11-18 최종 Ours(ausvp) + S · V 피험자 공통 (FiLM · 피험자 분류 끔)
                     ausvp_oraclesil) echo "$EU --mem branch --v_pool 2 --sv_enc 0 --oracle_sil $TGT/silhouettes32.npz";;   # 11-18 V-0 (진단 전용 상한): 최종 Ours(ausvp) + 정답 실루엣 지도(영상 10프레임 평균)를 V 메모리에
                     ausvp_oraclesil_catmean) echo "$EU --mem branch --v_pool 2 --sv_enc 0 --oracle_sil $W/prism_calib/vmaps/sil_catmean.npz";;   # 11-19 V-0 대조: 정답 범주의 평균 실루엣 지도만 (diag_vceil.py --maps_out)
                     ausvp_oraclesil_pred) echo "$EU --mem branch --v_pool 2 --sv_enc 0 --oracle_sil $W/prism_calib/vmaps/sil_pred.npz";;   # 11-19 V-0 대조: 범주 평균 + V1–V4 커널 릿지 예측 잔차 (피험자별, train은 5겹 밖 예측)
                     ausvp_siltmpl) echo "$EU --mem branch --v_pool 2 --sv_enc 0 --sil_tmpl $TGT/silhouettes32.npz";;   # 11-19 V-T: 최종 Ours(ausvp) + 범주 평균 실루엣을 Semantic 예측 범주 확률로 섞어 V 메모리에 (정답 안 씀)
                     ausvp_linpath) echo "$EU --mem branch --v_pool 2 --sv_enc 0 --lin_path 1 --lin_norm 1";;   # 11-19 C-1: 최종 Ours(ausvp) + Semantic 원 신호 선형 경로(프레임 평균 65,536픽셀 → 1024, 피험자별, 0 초기화, 입력 z-score)
                     base_vres_sharedsv_calib) echo "$EU --mem branch --v_pool 2 --v_res_basis ${VRB:-$TGT/vres_basis_k32.npz} --shared_sv 1 --film 0 --lam_subj 0 --calib 1 --calib_init ${CAL:-$W/prism_calib/s16/calib.pt}";;   # V 잔차 + Perceptive 개선안 둘 다
                     ausv) echo "$EU --mem branch --sv_enc 0";; auc) echo "$EU --mem branch --v_center 1";;
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
    --workers ${WK:-4} --polite ${POLITE:-65} --per_subject 1 --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 \
    --seed 1 --holdout $HOLD ${SAVE:+--save_steps $SAVE} $(flags $n) $X --resume --out_dir $M > $L 2>&1
  echo "=== 학습 끝 $n $(date +%H:%M:%S) | $(grep -E '^\[hold\]' $L | tail -1 | cut -c1-200)"
  [ $TEST = 1 ] && { grep -E 'Traceback|Error|out of memory' $L | tail -3; grep -oE 'step [0-9]+/[0-9]+.*' $L | tail -1 | cut -c1-160; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo; }
}
if [ $TEST = 1 ]; then for n in $RUNS; do one $n; done
else J=""   # 11-17: 공용 GPU 배려 — 동시에 MAXJ개까지, GPU 남은 메모리 MINFREE MiB 이상 · CPU 1분 부하 LOADMAX 미만일 때만 새로 띄움
  MAXJ=${MAXJ:-2}; MINFREE=${MINFREE:-20000}; LOADMAX=${LOADMAX:-40}
  gate() { while :; do nrun=$(jobs -rp | wc -l); free=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.free --format=csv,noheader,nounits); load=$(cut -d' ' -f1 /proc/loadavg)
    [ $nrun -lt $MAXJ ] && [ $free -ge $MINFREE ] && awk "BEGIN{exit !($load < $LOADMAX)}" && return; sleep 30; done; }
  for n in $RUNS; do gate; echo "=== 띄움 $n $(date +%H:%M:%S) | 도는 것 $nrun/$MAXJ · GPU 남은 ${free} MiB · 부하 $load"; one $n & J="$J $!"; sleep 120; done; wait $J
  echo "=== 요약 (검증 물체 기준)"; SET=branch OUT=$O LP=$LP RUNS="$RUNS" bash $PR/scripts/status_reg.sh | sed -n '/^요약/,$p'; fi
echo "=== BRANCH_DONE $(date +%H:%M:%S)"
