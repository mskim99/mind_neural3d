#!/bin/bash
# 토큰 prior 1단계 데이터 (11-09, Pro6000 GPU 1): prior 학습에 쓸 정답 토큰 · 조건 만들기 (토큰prior_설계.md, 사용자 11-09: PRISM 성분 조건 · 프레임 6장 · sub-0002로 조정)
#  0 디스크 확인 (dino_v1~5 약 20 GB 추정, 남은 용량 30 GB 미만이면 멈춤)
#  1 (뒤에서) 자극 영상 프레임 1~5 → TRELLIS 조건 토큰 <TGT>/dino_v1~5 (프레임 0은 이미 <TGT>/dino)
#  2 PRISM 2겹 교차 학습 (fig 설정, 7명, bs 14, 2만 스텝 = train 절반이라 fig 4만 스텝과 같은 바퀴 수) fold 0 · 1 동시에
#  3 안 본 train 예측: fold k 모델 → train에서 i % 2 == k 물체, 7명 → oof<k>/<sub>_keep/components.npz (tok_pool · cls_score, 토큰 파일은 안 씀)
#  4 fig 모델 test 예측 sub-0001 · sub-0002 (토큰 + tok_pool) → infer_fig/<sub>_keep
#  → PRIOR1_DATA_DONE. 다음은 run_prior1_tune.sh
# 사용: nohup bash run_prior1_data.sh > $W/logs/prism_prior1_data.log 2>&1 < /dev/null & disown
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
ENC=$W/weights/MinD-3D-pp/fMRI-Shape; PY=$W/venv/bin/python; P1=$W/prism_prior1; FIG=$W/prism_fig/model_final.pt
step() { echo "=== $1 $(date +%H:%M:%S)"; }
# 인자는 함수 위치 인자로 넘김: mg_guard.sh가 안에서 변수 v를 덮어써서 (11-09 dino_v 0개 원인) source 뒤에 $v를 쓰면 안 됨
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $P1; [ -f $FIG ] || { echo "fig 체크포인트 없음 $FIG"; exit 1; }

step "0 디스크"
df -h $W | tail -1; AV=$(df -BG --output=avail $W | tail -1 | tr -dc 0-9)
[ $AV -ge 30 ] || { echo "남은 용량 ${AV}G < 30G → 멈춤 (정리 후 다시)"; exit 1; }

step "1 프레임 1~5 토큰 (뒤에서, 로그 logs/prism_dino_v.log)"
( for vw in 1 2 3 4 5; do
    n=$(ls $TGT/dino_v$vw/*.npy 2>/dev/null | wc -l)
    [ $n -ge 1404 ] || trel dino --frames $TGT/frames --out $TGT/dino_v$vw --view $vw 2>&1 | grep -E "DINO_DONE|Error|error|dino"
    echo "dino_v$vw $(ls $TGT/dino_v$vw/*.npy 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  done ) > $W/logs/prism_dino_v.log 2>&1 &
DV=$!

step "2 PRISM 2겹 교차 학습 (fold 0 · 1 동시, 각 2만 스텝)"
TP=""
for k in 0 1; do
  M=$P1/fold$k
  if [ -f $M/model_final.pt ]; then echo "이미 있음 $M"; continue; fi
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --init_ckpt $INIT --bs 14 --max_step 20000 \
    --per_subject 1 --enc_dir $ENC --arch fig --sem_clip front --var_heads sr --var_mode feat --fold $k/2 --resume --out_dir $M > $W/logs/prism_prior1_fold$k.log 2>&1 &
  TP="$TP $!"; sleep 60                                                       # 두 번째 학습이 첫 번째 로딩과 겹치지 않게
done
[ -n "$TP" ] && wait $TP
for k in 0 1; do
  [ -f $P1/fold$k/model_final.pt ] || { echo "fold $k 학습 실패: $W/logs/prism_prior1_fold$k.log"; tail -20 $W/logs/prism_prior1_fold$k.log; exit 1; }
  echo "fold $k 끝 | $(grep -E '^\[train\]' $W/logs/prism_prior1_fold$k.log | head -1) | $(grep -E '^\[val\]' $W/logs/prism_prior1_fold$k.log | tail -1)"
done

step "3 안 본 train 예측 (fold k 모델 → i % 2 == k 물체, 7명)"
for k in 0 1; do
  [ $(ls $P1/oof$k/sub-*_keep/components.npz 2>/dev/null | wc -l) -ge 7 ] && { echo "이미 있음 oof$k"; continue; }
  $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $P1/fold$k/model_final.pt --out $P1/oof$k --split train --fold $k/2 \
    --perceptive keep --tok_pool 1 --save_tokens 0 | grep -E "범주|DONE|Error"
done

step "4 fig test 예측 sub-0001 · sub-0002 (토큰 + 요약)"
[ -f $P1/infer_fig/sub-0002_keep/components.npz ] && echo "이미 있음 infer_fig" || \
$PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $FIG --out $P1/infer_fig --subs sub-0001,sub-0002 --perceptive keep --tok_pool 1 | grep -E "범주|DONE|Error"

step "5 프레임 토큰 기다림"
wait $DV; grep -E "^dino_v|Error" $W/logs/prism_dino_v.log
N=$(ls $P1/oof*/sub-*_keep/components.npz 2>/dev/null | wc -l); echo "조건 파일 $N개 (예상 14 = 2겹 × 7명)"; df -h $W | tail -1; [ $N -eq 14 ] || { echo "조건 파일 수가 다름"; exit 1; }
for vw in 1 2 3 4 5; do [ $(ls $TGT/dino_v$vw/*.npy 2>/dev/null | wc -l) -ge 1404 ] || { echo "dino_v$vw 부족 → logs/prism_dino_v.log"; exit 1; }; done
step "PRIOR1_DATA_DONE"
