#!/bin/bash
# A안 2단계 (11-10, 5090 GPU 0): Semantic 안 이미지 임베딩 head 학습 → 예측 임베딩 → unCLIP → 외형(SLat)
#  학습 = fig 기준선과 같은 설정(arch fig · sem_clip front · var_heads sr · 7명 · bs 14 · 4만 스텝) + --lam_img 1 --lam_img_nce 1
#  정답 = 자극 프레임 0의 unCLIP 이미지 인코더 임베딩 ($TGT/clipimg). 원문에 없는 설정 — 사용자 확인 전 시험 (11-10 밤, 사용자 자율 진행 지시)
#  unCLIP 설정은 UC_PROMPT · UC_NL · UC_CFG (밝기 보정 결과로 고름)
# 사용: GPU=0 nohup bash run_img.sh > $W/logs/prism_img.log 2>&1 < /dev/null & disown
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; UPY=$W/venv_unclip/bin/python; ENC=$W/weights/MinD-3D-pp/fMRI-Shape
M=$W/prism_img; I=$M/infer; CK=$M/model_final.pt; Q=$PR/scripts/qual14_uids.txt; STEPS=${STEPS:-40000}
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $M; export HF_HUB_OFFLINE=1

step "1 정답 임베딩 (자극 프레임 0, 1,404개)"
$UPY $PR/unclip_tools.py embed --frames $TGT/frames --out $TGT/clipimg | grep -E "EMBED_DONE|Error"

step "2 학습 ($STEPS 스텝)"
if [ ! -f $CK ]; then
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step $STEPS --per_subject 1 --enc_dir $ENC \
    --arch fig --sem_clip front --var_heads sr --var_mode feat --lam_img 1 --lam_img_nce 1 --out_dir $M > $W/logs/prism_img_train.log 2>&1
fi
[ -f $CK ] || { echo "학습 실패"; tail -20 $W/logs/prism_img_train.log; exit 1; }
grep -E "^\[val\]" $W/logs/prism_img_train.log | tail -2

step "3 sub-0001 추론 · 임베딩 평가"
$PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I --subs sub-0001 --perceptive keep | grep -E "범주|DONE"
$PY $PR/eval_img.py --comp $I/sub-0001_keep/components.npz --clipimg $TGT/clipimg --train_list $DR/annotations/core_train_list.txt --json $M/eval_img_s1.json

step "4 예측 임베딩 → unCLIP → DINO 토큰 → 외형 (정성 14물체)"
[ -f $M/uc.env ] && source $M/uc.env && echo "unCLIP 설정 $M/uc.env: prompt=\"$UC_PROMPT\" nl=$UC_NL cfg=$UC_CFG pre=$UC_PRE"   # 밝기 진단 결과를 학습 중에 써 둘 수 있게
NORM=$($PY -c "import numpy as np,glob; print(np.mean([np.linalg.norm(np.load(f)) for f in glob.glob('$TGT/clipimg/*.npy')]))")
$PY -c "
import numpy as np; z=np.load('$I/sub-0001_keep/components.npz'); np.savez('$M/pred_img_s1.npz', **{str(k).split('/')[1]: v for k, v in zip(z['name'], z['img'])})"
$UPY $PR/unclip_tools.py gen --frames $TGT/frames --names $Q --out $M/img_pred --embeds $M/pred_img_s1.npz --norm $NORM \
  --prompt "${UC_PROMPT:-}" --noise_level ${UC_NL:-0} --cfg ${UC_CFG:-10} | grep -E "DONE|Error"
trel dino --frames $M/img_pred --out $M/tok_pred --rembg ${UC_REMBG:-1} | grep -E "DONE|Error"
trel gscheck --pred $I/sub-0001_keep/tokens --slat_pred $M/tok_pred --out $M/quick/unclip_pred --names $Q | grep -E "GSCHECK_DONE|Error|없음"
echo "  unclip_pred 색: $($PY $PR/scripts/gs_summary.py $M/quick/unclip_pred/gscheck.jsonl)"
step "IMG_DONE"
