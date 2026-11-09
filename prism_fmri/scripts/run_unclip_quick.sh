#!/bin/bash
# A안 1단계 — 학습 없는 상한 (11-09, 5090 GPU 0, 추론만): 정답 자극 프레임 0의 CLIP 이미지 임베딩 → SD 2.1 unCLIP → 이미지 → DINO 토큰(rembg)
#  → TRELLIS 외형(SLat)에만, 형상(SS)은 fig 예측 토큰. 정성 14물체(qual14) × sub-0001, Gaussian 렌더(gscheck)로 비교
#  비교 5줄: pred(예측 그대로) · text(텍스트판 "a {예측 범주}", 지금 방법) · unclip_gt(이번 상한) · unclip_full(unCLIP 이미지로 TRELLIS 두 단계 다) · gt(외형 = 정답 프레임 토큰)
# 사용: nohup bash run_unclip_quick.sh > $W/logs/prism_unclip_quick.log 2>&1 < /dev/null & disown
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; UPY=$W/venv_unclip/bin/python; O=$W/prism_unclip; I=$O/infer/sub-0001_keep; Q=$PR/scripts/qual14_uids.txt
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $O

step "0 fig sub-0001 추론 (예측 토큰 · 범주)"
[ -f $I/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_fig/model_final.pt --out $O/infer --subs sub-0001 --perceptive keep | grep -E "범주|DONE"
$PY $PR/text_prompts.py --comp $I/components.npz --out $O/prompt_s1.json

step "1 임베딩 공간 확인 (unCLIP 인코더 vs OpenCLIP ViT-H-14 laion2b)"
HF_HUB_OFFLINE=1 $UPY $PR/unclip_tools.py embcheck --frames $TGT/frames --names $Q | grep -E "EMBCHECK|차원|Error"

step "2 unCLIP 이미지 (정답 프레임 임베딩)"
HF_HUB_OFFLINE=1 $UPY $PR/unclip_tools.py gen --frames $TGT/frames --names $Q --out $O/img_gt | grep -E "unclip 1/|DONE|Error"
step "3 생성 이미지 → DINO 토큰 (rembg)"
trel dino --frames $O/img_gt --out $O/tok_gt --rembg 1 | grep -E "cond shape|DONE|Error"

step "4 Gaussian 렌더 5줄"
run() { v=$1; shift; [ $(cat $O/quick/$v/gscheck.jsonl 2>/dev/null | wc -l) -ge 14 ] && return
  trel gscheck --out $O/quick/$v --names $Q "$@" | grep -E "텍스트판|GSCHECK_DONE|Error|error|없음"; }
run pred        --pred $I/tokens
run text        --pred $I/tokens --slat_text $O/prompt_s1.json
run unclip_gt   --pred $I/tokens --slat_pred $O/tok_gt
run unclip_full --pred $O/tok_gt
run gt          --pred $I/tokens --slat_pred $TGT/dino
for v in pred text unclip_gt unclip_full gt; do echo "  $v 색: $($PY $PR/scripts/gs_summary.py $O/quick/$v/gscheck.jsonl)"; done

step "5 몽타주"
$UPY $PR/scripts/montage_unclip.py --root $O --frames $TGT/frames --names $Q --prompt $O/prompt_s1.json --out $O/비교_unclip.png
step "UNCLIP_QUICK_DONE"
