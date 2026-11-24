#!/bin/bash
# P-4 (11-24, 5090): 최종 Ours 7명 모델(재학습 없음)에 처음 보는 4번을 같은 물체 짝 0 · 100 · 300 · 1,300개로 1번 자리에 붙여 3D 생성 · 평가 (기준 방식 = 인코더 특징 전체의 선형 변환, test 104개는 안 씀)
#  세트마다 GLB("a {예측 범주}" 텍스트판) → 방향 보정 → 평가 기준(run_dir_eval.sh), 한 세트씩
# 사용 (절대 경로로): nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_p4.sh > $W/logs/p4.log 2>&1 < /dev/null &
GPU=0 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
cd $W/run; PY=$W/venv/bin/python; O=$W/prism_p4; mkdir -p $O; NS=${NS:-0,100,300,1300}
if [ ! -f $O/DONE ]; then wait_gpu_free p4_align; echo "=== p4_align $(date +%H:%M:%S)"
  $PY $PR/align_infer.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_full/ausvp/model_final.pt --subs sub-0001 --fmri_subs sub-0004 --space feat --fmap full --fit mean --ns $NS --save $NS --save_tok 1 --out $O && touch $O/DONE; fi
[ -f $O/DONE ] || { echo "p4_align 실패"; exit 1; }
for n in ${NS//,/ }; do
  wait_gpu_free p4_n$n
  GPU=0 SETS="p4_s04_n${n}_as01=$O/n$n/sub-0004_as01_keep" bash $PR/scripts/run_dir_eval.sh
done
echo "=== P4_DONE $(date +%H:%M:%S)"
