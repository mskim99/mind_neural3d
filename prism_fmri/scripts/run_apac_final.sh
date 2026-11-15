#!/bin/bash
# 최종 Ours(ausvp, 1,300개 · 7명) AP · APAC (11-15, 사용자) — 형태 지표만 (FPD · CD · EMD)
#  기존 Ours AP/APAC와 같은 방식: Subject 1 갈래에 sub-0009(AP, core test 104) · sub-0011(APAC, 220) fMRI 입력 → 텍스트판 GLB("a {예측 범주}")
#  → 방향 보정(12방위 렌더 → yawsel → 회전, run_sub_eval.sh와 같은 명령) → fe_shape
#  준비: $DR/sub-0009 · sub-0011 (npy_data), $DR/annotations/apac_sub0011_list.txt, APAC GT 점구름 $FE/gt_pc/P2 (11-15 올림)
#  → $FE/out/yaw_fullausvp_{AP,APAC220}/values.txt
# 사용: GPU=0 X=AP nohup bash run_apac_final.sh > $W/logs/apac_final_AP.log 2>&1 < /dev/null & disown   (APAC는 GPU=1 X=APAC)
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
source $(dirname $0)/fe_shape.sh
X=${X:?AP 또는 APAC}; FE=/home/oem/jhj/fair_eval/fair_eval; PY=$W/venv/bin/python; TM=microsoft/TRELLIS-text-base
CK=${CK:-$W/prism_full/ausvp/model_final.pt}; TAG=${TAG:-fullausvp}; I=$W/prism_full/apac_$TAG
declare -A FS=([AP]=sub-0009 [APAC]=sub-0011) LS=([AP]=$DR/annotations/core_test_list.txt [APAC]=$DR/annotations/apac_sub0011_list.txt) NN=([AP]=104 [APAC]=220) TG=([AP]=AP [APAC]=APAC220)
export FE_PY=/home/oem/jhj/fair_eval/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=/home/oem/jhj/fair_eval/torch_home FE_WORKERS=${FE_WORKERS:-6}
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
S=${FS[$X]}; N=${NN[$X]}; L=${LS[$X]}; d=$I/${S}_as01_keep; Y=$W/prism_yaw/${TAG}_${TG[$X]}
[ $(find $DR/$S/npy_data -name '*.npy' | wc -l) -ge $N ] && [ -f $L ] || { echo "자료 없음 $S / $L"; exit 1; }
cd $W/run; mkdir -p $I $Y
echo "=== $X 추론 $(date +%H:%M:%S)"
[ -f $d/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I --subs sub-0001 --fmri_sub $S --list $L --perceptive keep | grep -E "범주|DONE|Error"
$PY $PR/text_prompts.py --comp $d/components.npz --out $I/prompt_$X.json | tail -1
[ $(ls $d/glb/*/*.glb 2>/dev/null | wc -l) -ge $N ] || trel gen --pred $d/tokens --slat_text $I/prompt_$X.json --text_model $TM --out $d/glb | grep -E "GEN_DONE|Error|없음"
echo "T|$X|GLB $(ls $d/glb/*/*.glb 2>/dev/null | wc -l)/$N|$(date +%H:%M:%S)"
if [ ! -f $Y/yawsel.json ]; then                          # 방향 보정 (run_sub_eval.sh post와 같음)
  ls $d/glb/*/*.glb > $Y/list.txt
  $FE_BLENDER -b -t 4 -P $FE/scripts/render.py -- --list $Y/list.txt --out_root $Y/render --res 512 --light 8000 --camera_dist 1.5 --fov 56 \
    --views $(seq -s, 0 30 330 | sed 's/\([0-9]*\)/\1:30/g') --metallic0 2>&1 | grep -E "FAIL" | head -5
  echo "R|$X|렌더 $(find $Y/render -name '*.png' | wc -l)/$((N * 12))|$(date +%H:%M:%S)"
  trel yawsel --frames $Y/render --pred $d/tokens --out $Y --nview 12 | grep -E "YAWSEL_DONE|Error|없음"
fi
[ $(ls $Y/glb/*/*.glb 2>/dev/null | wc -l) -ge $N ] || $PY $PR/yaw_apply.py --glb $d/glb --sel $Y/yawsel.json --out $Y/glb | grep YAWAPPLY
fe_shape yaw_${TAG}_${TG[$X]} $Y/glb glb $L
echo "=== APAC_FINAL_DONE $X $(date +%H:%M:%S)"
