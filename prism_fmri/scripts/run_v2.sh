#!/bin/bash
# V-2 (11-20, 5090): 최종 Ours(7명 학습) 피험자별로 메시 비율 조정 6조건을 만들고 형태 지표(CD · EMD · FPD)만 평가 번들로 다시 잼 (텍스처 건너뜀)
# 사용: GPU=0 NS="01 06" nohup bash $W/mind_neural3d/prism_fmri/scripts/run_v2.sh > $W/logs/v2.log 2>&1 < /dev/null & disown
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
F=/home/oem/jhj/fair_eval; V=$F/fair_eval; export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home FE_SKIP_TEX=1 FE_WORKERS=6
for s in ${NS:-01 06}; do
  O=$W/prism_v2/s$s
  [ -f $O/v2.json ] || $FE_PY $PR/v2_rescale.py --glb $W/prism_yaw/fullausvp_s$s/glb --render $V/out/yaw_fullausvp_s$s/render_R2 --comp $W/prism_full/eval_ausvp/infer/s$s/sub-00${s}_keep/components.npz \
    --sil $W/data/shape_targets/view_sil.npz --data_root $DR --gt_pc $V/gt_pc/P2 --out $O | cut -c1-900
  for c in same gt3 gtsil vpred catmean other; do
    E=v2_${c}_s$s
    [ -f $V/out/$E/values.txt ] || bash $V/run_eval.sh $E $O/$c glb $GPU > $F/logs/eval_$E.log 2>&1
    echo "V2|s$s|$c | $(grep -E '^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(date +%H:%M:%S)"
  done
done
echo "=== V2_DONE $(date +%H:%M:%S)"
