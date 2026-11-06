#!/bin/bash
# 7명 기준선(prism_shape_ps7 final) 지표 평가 (11-05, Pro6000 GPU 1). 사용: bash run_ps7_eval.sh   로그 logs/prism_ps7_eval.log
#  피험자마다: 토큰(prism_infer_ps7, 이미 있음) → TRELLIS GLB → MinD-3D++ 논문 값을 재현한 평가 설정(spp_eval/run_spp.sh)으로 8지표
#  GLB 생성과 평가를 겹쳐 돌림: 피험자 s의 GLB가 끝나면 s 평가를 뒤에서 시작하고 바로 다음 피험자 GLB로 (평가는 한 번에 하나)
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
I=$W/prism_infer_ps7; EV=""
for s in ${SUBS//,/ }; do
  echo "=== GLB $s $(date +%H:%M:%S)"
  (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py gen --pred $I/${s}_keep/tokens --out $I/${s}_keep/glb) | grep -E "GEN_DONE|Error"
  echo "GLB $s $(ls $I/${s}_keep/glb/*/*.glb 2>/dev/null | wc -l)개"
  [ -n "$EV" ] && wait $EV
  (cd $SPP_ROOT && bash run_spp.sh ps7_$s $I/${s}_keep/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_ps7_$s.log 2>&1; echo "=== EVAL $s 끝 $(date +%H:%M:%S) | $(tail -2 $W/logs/spp_ps7_$s.log | head -1)") &
  EV=$!
done
wait
echo "=== 7명 지표"
for s in ${SUBS//,/ }; do echo "$s $(cat $SPP_ROOT/out/ps7_$s/values.txt 2>/dev/null)"; done
echo "PS7_EVAL_DONE $(date +%H:%M:%S)"
