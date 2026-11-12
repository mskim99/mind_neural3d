#!/bin/bash
# 판정 2 (11-12): 갈래 끄기 · 바꿔 끼우기 (swap_test.py), 7명 × test 104, 토큰만
#  GPU=0 MODELS="g0"            → $W/prism_swap/g0/swap.json  (+ qual14 물체 S_diff · V_diff · V_same 토큰 → GLB)
#  GPU=1 MODELS="onlyS onlyV"   → $W/prism_swap/<이름>/swap.json
# 사용: GPU=0 MODELS=g0 nohup bash run_swap.sh > $W/logs/swap_g0.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
PY=$W/venv/bin/python; O=$W/prism_swap; EX=$PR/scripts/qual14_uids.txt
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $O
for m in ${MODELS:-g0}; do
  case $m in
    g0) CK=$W/prism_vshape/vshape_g0/model_final.pt; X="--ex $EX";;
    *)  CK=$W/prism_abl/$m/model_final.pt; X="";;
  esac
  echo "=== $m $(date +%H:%M:%S)"
  [ -f $O/$m/swap.json ] || $PY $PR/swap_test.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --ckpt $CK --out $O/$m $X | grep -vE "Warning|warn"
  if [ -n "$X" ]; then
    for c in S_diff V_diff V_same; do
      [ $(ls $O/$m/ex_glb/$c/*/*.glb 2>/dev/null | wc -l) -ge 14 ] || trel gen --pred $O/$m/ex_tokens/$c --out $O/$m/ex_glb/$c | grep -E "GEN_DONE|Error"
    done
  fi
done
echo "SWAP_ALL_DONE $(date +%H:%M:%S)"
