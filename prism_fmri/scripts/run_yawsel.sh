#!/bin/bash
# GT 없이 방향 고르기 시험 (11-11, 사용자: "일단 시도", Pro6000 GPU 1) — 설계 `.보고 자료/261112_9차보고/PRISM_분리/분리·캡션_결과.md` 5절
#  1 생성 GLB를 자극 카메라(거리 1.5 · FOV 56 · 고도 30, 평가 R2 뷰 0과 같음)로 방위 0 · 30 · … · 330 렌더 (512², 공정평가 render.py)
#  2 trellis_tools yawsel: 렌더마다 TRELLIS DINO 토큰 → 예측 토큰(자극 프레임 0 흉내)과 가장 비슷한 방위 → 고칠 회전
#  3 yaw_compare: GT로 찾은 최선 각도(prism_sep/yaw/*.json)와 비교 — 정답 토큰(gttok)에서 먼저 되는지 보고, 되면 Ours(g0)
#  → $W/prism_sep/yawsel/<이름>/yawsel.json, 로그 YAWCMP|… (YAWSEL_ALL_DONE)
# 사용: nohup bash run_yawsel.sh > $W/logs/prism_yawsel.log 2>&1 < /dev/null & disown      (SETS="gttok"처럼 고를 수 있음)
source $(dirname $0)/common.sh || exit 1
SETS=${SETS:-"gttok g0"}
FE=$W/fair_eval; PY=$W/venv/bin/python; B=$W/tools/blender-4.2.3-linux-x64/blender; O=$W/prism_sep/yawsel
declare -A GLB=([gttok]=$W/prism_sep/glb_gttok [g0]=$W/prism_vshape/infer/vshape_g0/sub-0001_keep/glb)
declare -A TOK=([gttok]=$W/prism_sep/gt_tok [g0]=$W/prism_vshape/infer/vshape_g0/sub-0001_keep/tokens)
declare -A YJ=([gttok]=$W/prism_sep/yaw/gt_tok.json [g0]=$W/prism_sep/yaw/ours_g0.json)
VIEWS=$(seq -s, 0 30 330 | sed 's/\([0-9]*\)/\1:30/g')
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
for s in $SETS; do
  [ -d ${GLB[$s]} ] && [ -d ${TOK[$s]} ] && [ -f ${YJ[$s]} ] || { echo "없음 $s: ${GLB[$s]} · ${TOK[$s]} · ${YJ[$s]}"; continue; }
  mkdir -p $O/$s; awk -F/ -v d=${GLB[$s]} 'NF{u=$2; sub(/\r$/,"",u); print d"/"u"/"u".glb"}' $FE/core_test_list.txt > $O/$s/list.txt
  echo "=== $s 렌더 $(date +%H:%M:%S)"
  $B -b -t 4 -P $FE/scripts/render.py -- --list $O/$s/list.txt --out_root $O/$s/render --res 512 --light 8000 --camera_dist 1.5 --fov 56 --views $VIEWS --metallic0 2>&1 | grep -E "FAIL|RENDER_DONE"
  echo "=== $s 방향 고르기 $(date +%H:%M:%S)"
  trel yawsel --frames $O/$s/render --pred ${TOK[$s]} --out $O/$s --nview 12 | grep -E "yawsel|YAWSEL_DONE|Error|없음"
  $PY $PR/yaw_compare.py --yaw ${YJ[$s]} --sel $O/$s/yawsel.json --name $s
done
echo "=== YAWSEL_ALL_DONE $(date +%H:%M:%S)"
