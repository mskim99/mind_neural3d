#!/bin/bash
# B안 렌더 보정 (11-09, 5090 GPU 1): train 물체 범주마다 2개(26개) 실루엣을 yaw 4 × fov 21로 렌더 → 자극 프레임 0과 IoU
# 사용: nohup bash run_b_calib.sh > $W/logs/prism_b_calib.log 2>&1 < /dev/null & disown
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; BL=$W/tools/blender-4.2.3-linux-x64/blender; SN=$W/data/shapenet_b; O=$W/prism_b/calib
mkdir -p $O; cd $W/run
$PY $PR/calib_b.py pick --objects $SN/objects_b.txt --train $DR/annotations/core_train_list.txt --per 2 --out $O/calib_list.txt
FOVS=$(seq -s, 20 2 60)
$BL -b -P $PR/render_b.py -- --mode calib --root $SN --list $O/calib_list.txt --out $O/sil --fovs $FOVS --yaws 0,90,180,270 --res 256 2>&1 | grep -E "^\[calib|LOAD_FAIL|RENDER_B_DONE|Error"
$PY $PR/calib_b.py score --calib $O/sil --frames $TGT/frames
echo B_CALIB_DONE
