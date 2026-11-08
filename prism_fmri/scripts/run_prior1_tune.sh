#!/bin/bash
# 토큰 prior 1단계 학습 · 조정 (11-09, Pro6000 GPU 1): run_prior1_data.sh 다음
#  1 prior 학습 (prior_token.py, 조건 = 안 본 train 예측 14파일, 정답 = 프레임 0~5 토큰, 6만 스텝)
#  2 조정용 물체 = prior 학습에서 뺀 train 물체(prior/val_objs.txt) 14개, 피험자 sub-0002. test 물체는 안 씀 (sub-0001 test 104 보고와 겹치지 않게)
#    조건 = 그 물체를 안 본 fold 모델의 sub-0002 예측 (fold 0 · 1 합침). cfg 1 · 2 · 4 생성 (+ 복사 검사)
#    비교 = 예측 그대로 · 검색(자기 자신 제외) · 정답(그 물체 프레임 0 토큰)
#  3 형상(SS) = 그 예측 토큰, 외형(SLat)만 바꿈 → Gaussian 렌더 → prior1_tune.tar (PRIOR1_TUNE_DONE)
# 사용: nohup bash run_prior1_tune.sh > $W/logs/prism_prior1_tune.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; P1=$W/prism_prior1; S2=sub-0002; T=$P1/tune; Q=$T/tune14.txt
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; [ -f $P1/fold1/model_final.pt ] || { echo "fold 모델 없음 (run_prior1_data.sh 먼저)"; exit 1; }
OOF=$(ls $P1/oof*/sub-*_keep/components.npz | paste -sd,); NO=$(echo $OOF | tr , '\n' | wc -l)
echo "조건 파일 ${NO}개 (예상 14)"; [ $NO -eq 14 ] || { echo "조건 파일 수가 다름 → run_prior1_data.sh 로그 확인"; exit 1; }

step "1 prior 학습 (6만 스텝, 로그 logs/prism_prior1_train.log)"
if [ -f $P1/prior/prior_final.pt ]; then echo "이미 있음"; else
  $PY $PR/prior_token.py train --tgt $TGT --oof $OOF --out $P1/prior --steps 60000 --bs 32 --resume > $W/logs/prism_prior1_train.log 2>&1
fi
[ -f $P1/prior/prior_final.pt ] || { echo "prior 학습 실패"; tail -20 $W/logs/prism_prior1_train.log; exit 1; }
echo "학습 끝 | $(grep -E '^\[val\]' $W/logs/prism_prior1_train.log | tail -1)"

step "2 조정용 물체 (prior 학습에서 뺀 train 14개, sub-0002) 예측 · 생성 · 검색"
mkdir -p $T/pred; head -14 $P1/prior/val_objs.txt > $Q
for k in 0 1; do
  $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $P1/fold$k/model_final.pt --out $T/oof$k --subs $S2 --split train --fold $k/2 --names $Q \
    --perceptive keep --tok_pool 1 | grep -E "범주|DONE|Error"
  ln -sf $T/oof$k/${S2}_keep/tokens/*.npy $T/pred/
done
$PY $PR/merge_comp.py $T/comp.npz $T/oof0/${S2}_keep/components.npz $T/oof1/${S2}_keep/components.npz
echo "조정 물체 예측 토큰 $(ls $T/pred/*.npy | wc -l)개 (예상 14)"
for c in 1 2 4; do
  $PY $PR/prior_token.py sample --ckpt $P1/prior/prior_final.pt --comp $T/comp.npz --out $T/gen_cfg$c --cfg $c --names $Q --tgt $TGT --data_root $DR | grep -E "생성|COPY|Error"
done
$PY $PR/slat_retrieve.py --pred $T/pred --comp $T/comp.npz --tgt $TGT --data_root $DR --out $T/retr --exclude_self 1 | grep -E "SLATRET|Error"

step "3 빠른 확인 (조정 물체 14)"
run() {   # 이름 · trellis_tools 인자
  v=$1; shift; q=$T/quick/$v; [ $(cat $q/gscheck.jsonl 2>/dev/null | wc -l) -ge 14 ] && return
  [ -d $q ] && mkdir -p $T/quick_bad && mv $q $T/quick_bad/${v}_$(date +%s)   # 끊긴 결과는 옆으로 치움 (rm 안 씀)
  trel gscheck --pred $T/pred --out $q --names $Q "$@" | grep -E "GSCHECK_DONE|Error|없음"
}
run pred
run retr --slat_pred $T/retr
run gt --slat_pred $TGT/dino
for c in 1 2 4; do run cfg$c --slat_pred $T/gen_cfg$c; done
for d in $T/quick/*/; do
  echo "  $(basename $d) 색: $($PY $PR/scripts/gs_summary.py ${d}gscheck.jsonl)"
done
cd $T && tar cf prior1_tune.tar tune14.txt quick/*/*.png quick/*/gscheck.jsonl gen_cfg*/copy_check.json retr/slat_map.json -C $P1 prior/train_log.csv && echo "PACK $(tar tf prior1_tune.tar | grep -c png)개 (예상 84)"
step "PRIOR1_TUNE_DONE"
