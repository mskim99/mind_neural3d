#!/bin/bash
# V-5 (11-24, Pro6000 GPU 1, 학습 없음): 자극 저수준 외형을 범주와 별개로 fMRI에서 읽나 — 피험자 1 ~ 8, 입력 3가지 (v5_read.py)
#  특징은 로컬에서 만든 $W/v5_feat.npz (v5_feat.py, 영상 10프레임 평균). 결과 $W/prism_v5/{read.json, pred_*.npz}, 묶음 $W/prism_v5/v5_pack.tar
#  V-4 학습과 같은 GPU에서 같이 돌아도 됨 (우리 작업끼리, 인코더 추론 + 커널 릿지만)
# 사용: GPU=1 nohup bash run_v5.sh > $W/logs/v5.log 2>&1 < /dev/null & disown   (진행: watch -n 1 'grep -aE "^\[sub|범주 안|DONE|Error" /home/oem/jhj/MinD-3D++/logs/v5.log | tail -12')
SD=$(cd "$(dirname "$0")" && pwd); source $SD/common.sh || exit 1
O=$W/prism_v5; CK=$W/prism_full/ausvp/model_final.pt; FT=$W/v5_feat.npz; mkdir -p $O
echo "=== V-5 시작 $(date '+%m-%d %H:%M') | 코드 $(cd $PR && git log --oneline -1 | cut -c1-50) | 특징 $([ -f $FT ] && md5sum < $FT | cut -c1-8 || echo 없음) | 모델 $(md5sum < $CK | cut -c1-8)"
[ -f $FT ] || { echo "특징 파일 없음 $FT — 멈춤"; exit 1; }
wait_gpu_free "V-5 읽기"
cd $W/run && $W/venv/bin/python $PR/v5_read.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --feat $FT --out $O \
  --subs sub-0001,sub-0002,sub-0003,sub-0004,sub-0005,sub-0006,sub-0007,sub-0008
(cd $W && tar cf $O/v5_pack.tar prism_v5/read.json $(cd $W && ls prism_v5/pred_*.npz) logs/v5.log) && echo "PACK $O/v5_pack.tar $(du -h $O/v5_pack.tar | cut -f1)"
echo "=== V5_DONE $(date +%H:%M:%S)"
