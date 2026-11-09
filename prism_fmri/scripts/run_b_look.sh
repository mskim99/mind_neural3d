#!/bin/bash
# B안 렌더 조명 고르기 (11-09, 5090 GPU 1): calib 26개 × (월드 세기 × 해 세기) → calib_look_b.py로 자극 프레임 0과 색 비교
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; BL=$W/tools/blender-4.2.3-linux-x64/blender; SN=$W/data/shapenet_b; O=$W/prism_b/calib; cd $W/run
D=""
for L in ${LIGHTS:-0.5 1.0 1.5}; do for S in ${SUNS:-0 1 2}; do
  o=$O/look_L${L}_S${S}; D="$D,$o"
  $BL -b -P $PR/render_b.py -- --mode render --root $SN --list $O/calib_list.txt --out $o --fov 56 --yaw 0 --light $L --sun $S 2>&1 | grep -E "Error|LOAD_FAIL"
done; done
$PY $PR/calib_look_b.py --frames $TGT/frames --dirs ${D#,}
echo B_LOOK_DONE
