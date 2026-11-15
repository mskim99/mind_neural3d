#!/bin/bash
# Ablation 한 세트(행 · 피험자) 수리 (11-16): GLB가 104개 안 된 채 넘어갔거나 렌더 · 평가가 3번 실패한 세트를 처음부터 다시 (이미 된 것은 건너뜀)
#  GLB 생성 OOM(cudaMalloc)으로 noS s02가 78/104에서 멈춘 채 평가로 넘어간 일 → GLB 104개가 될 때까지(최대 3번) 채운 뒤 run_abl_rows.sh post와 같은 절차
# 사용: GPU=1 N=noS S=02 bash fix_abl_set.sh
W=/home/oem/jhj/MinD-3D++; SC=$W/mind_neural3d/prism_fmri/scripts; O=$W/prism_full_abl; F=/home/oem/jhj/fair_eval; V=$F/fair_eval; Y=$W/prism_yaw
N=${N:?}; s=${S:?}; GPU=${GPU:?}; source $SC/common.sh >/dev/null || exit 1; export CUDA_VISIBLE_DEVICES=$GPU
PY=$W/venv/bin/python; TM=microsoft/TRELLIS-text-base; SS=sub-00$s; C=$O/eval_$N; I=$C/infer/s$s; g=$C/text/s$s/glb; Yd=$Y/abl_${N}_s$s; E=yaw_abl_${N}_s$s
export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
freewait() { until [ "$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 30; done; }
cd $W/run
for t in 1 2 3; do
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] && break
  freewait 22000; trel gen --pred $I/${SS}_keep/tokens --slat_text $C/text/s$s/prompt.json --text_model $TM --out $g | grep -E "GEN_DONE|Error|없음"
done
k=$(ls $g/*/*.glb 2>/dev/null | wc -l); echo "T|$N s$s|GLB $k/104 (수리)|$(date +%H:%M:%S)"; [ $k -ge 104 ] || { echo "GLB 부족 → 멈춤"; exit 1; }
mkdir -p $Yd
for t in 1 2 3; do
  [ -f $Yd/yawsel.json ] && break
  ls $g/*/*.glb > $Yd/list.txt
  $FE_BLENDER -b -t 4 -P $V/scripts/render.py -- --list $Yd/list.txt --out_root $Yd/render --res 512 --light 8000 --camera_dist 1.5 --fov 56 \
    --views $(seq -s, 0 30 330 | sed 's/\([0-9]*\)/\1:30/g') --metallic0 2>&1 | grep -E "FAIL" | head -3
  k=$(find $Yd/render -name '*.png' | wc -l); [ $k -ge 1248 ] || { echo "R!|$N s$s 방향 렌더 $k/1248 → 다시 ($t)"; continue; }
  freewait 6000; trel yawsel --frames $Yd/render --pred $I/${SS}_keep/tokens --out $Yd --nview 12 | grep -E "YAWSEL_DONE|Error|없음"
done
[ $(ls $Yd/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/yaw_apply.py --glb $g --sel $Yd/yawsel.json --out $Yd/glb | grep YAWAPPLY
for t in 1 2 3; do
  grep -q "R2|LPIPS" $V/out/$E/values.txt 2>/dev/null && break
  [ $t -gt 1 ] && freewait 22000
  bash $V/run_eval.sh $E $Yd/glb glb $GPU > $F/logs/eval_$E.log 2>&1
  grep -q "R2|LPIPS" $V/out/$E/values.txt 2>/dev/null || echo "E!|$N s$s 외형 지표 없음 → 다시 ($t)"
done
echo "CV|$N s$s | $(grep -E '^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | awk '{printf "%s ", $2}') | $(date +%H:%M:%S) (수리)"
