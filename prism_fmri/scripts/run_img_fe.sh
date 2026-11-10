#!/bin/bash
# A안 공정평가 (11-10, 4090 인계 "잠정 기준 적용"): prism_img10k sub-0001, test 104 전부 → glb 두 세트 → 공정평가 잠정 번들
#  unclip: 형상 = 예측 토큰, 외형 = 예측 이미지 임베딩 → unCLIP(감마 0.5) → DINO(rembg)
#  text  : 형상 = 예측 토큰, 외형 = TRELLIS 텍스트판 "a {예측 범주}" (지금 방법, 대조)
# 사용: nohup bash run_img_fe.sh > $W/logs/prism_img_fe.log 2>&1 < /dev/null & disown
GPU=1 source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; UPY=$W/venv_unclip/bin/python; INF=$W/prism_img10k/infer/sub-0001_keep; O=$W/prism_img10k/fe104
FE=/home/oem/jhj/fair_eval
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
step() { echo "=== $1 $(date +%H:%M:%S)"; }
cd $W/run; mkdir -p $O; export HF_HUB_OFFLINE=1
ls $INF/tokens | sed 's/\.npy$//' > $O/names.txt
echo "물체 $(wc -l < $O/names.txt)개, 공정평가 목록과 겹침 $(cut -d/ -f2 $FE/fair_eval/core_test_list.txt | tr -d '\r' | grep -cxFf $O/names.txt)"
step "1 unCLIP 이미지 · DINO"
NORM=$($PY -c "import numpy as np,glob; print(np.mean([np.linalg.norm(np.load(f)) for f in glob.glob('$TGT/clipimg/*.npy')]))")
$PY -c "import numpy as np; z=np.load('$INF/components.npz'); np.savez('$O/pred_img.npz', **{str(k).split('/')[1]: v for k, v in zip(z['name'], z['img'])})"
$PY $PR/text_prompts.py --comp $INF/components.npz --out $O/prompt.json
CUDA_VISIBLE_DEVICES=1 $UPY $PR/unclip_tools.py gen --frames $TGT/frames --names $O/names.txt --out $O/img --embeds $O/pred_img.npz --norm $NORM --gamma 0.5 | grep -E "DONE|Error"
CUDA_VISIBLE_DEVICES=1 trel dino --frames $O/img --out $O/tok --rembg 1 | grep -E "DONE|Error"
step "2 glb 두 세트 (unclip GPU 1, text GPU 0)"
(CUDA_VISIBLE_DEVICES=1 trel gen --pred $INF/tokens --slat_pred $O/tok --out $O/glb_unclip | grep -E "GEN_DONE|Error|없음") &
(CUDA_VISIBLE_DEVICES=0 trel gen --pred $INF/tokens --slat_text $O/prompt.json --out $O/glb_text | grep -E "GEN_DONE|Error|없음") &
wait
echo "glb: unclip $(ls $O/glb_unclip/*/*.glb | wc -l) · text $(ls $O/glb_text/*/*.glb | wc -l)"
step "3 공정평가 (잠정 R2 · P2 · FPD)"
bash $FE/run_check.sh A_unclip_s1 $O/glb_unclip glb 1 > $FE/logs/A_unclip_s1.log 2>&1 &
bash $FE/run_check.sh A_text_s1 $O/glb_text glb 0 > $FE/logs/A_text_s1.log 2>&1 &
wait
tail -9 $FE/logs/A_unclip_s1.log; tail -9 $FE/logs/A_text_s1.log
step "IMG_FE_DONE"
