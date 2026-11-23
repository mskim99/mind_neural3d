#!/bin/bash
# 11-23: 7명 모델(prism_full/ausvp)에 AP 피험자 9번을 같은 물체 짝으로 붙이기. 9번은 test 104개만 봐서 test 물체를 K겹으로 나눠 맞춤 (align_infer.py --cv)
#  1) 범주 정확도: 학습 피험자 7명 각각에게, 2 · 4 · 13겹(짝 약 52 · 78 · 96개) + 그대로(짝 0) + 7명 점수 평균
#  2) 3D: 1번에게 붙인 것 3세트(그대로 / 4겹 / 13겹)를 조건 토큰까지 저장 → GLB → 방향 보정 → 평가 기준 (run_dir_eval.sh)
# 사용 (절대 경로로): bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_align_ap.sh
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd $W/run; PY=$W/venv/bin/python; CK=$W/prism_full/ausvp/model_final.pt; O=$W/prism_align7; mkdir -p $O
S7=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008; B=${B:-sub-0009}
AL="$PY $PR/align_infer.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --fmri_subs $B --space feat --fmap full --fit mean --cv 2,4,13"
if [ ! -f $O/ap_cls/DONE ]; then wait_gpu_free ap_cls; echo "=== ap_cls $(date +%H:%M:%S)"; $AL --subs $S7 --save 0,2,4,13 --save_tok 0 --out $O/ap_cls && touch $O/ap_cls/DONE; fi
$PY $PR/align_ens.py $O | grep -E "ap_cls|ENS"
if [ ! -f $O/ap_3d/DONE ]; then wait_gpu_free ap_3d; echo "=== ap_3d $(date +%H:%M:%S)"; $AL --subs sub-0001 --save 0,4,13 --save_tok 1 --out $O/ap_3d && touch $O/ap_3d/DONE; fi
for n in 0 78 96; do   # 한 세트씩 (생성과 평가가 GPU에서 겹치지 않게)
  wait_gpu_free dir_eval_n$n
  GPU=0 SETS="ap9_n${n}_as01=$O/ap_3d/n$n/${B}_as01_keep" bash $PR/scripts/run_dir_eval.sh
done
echo "=== ALIGN_AP_DONE $(date +%H:%M:%S)"
