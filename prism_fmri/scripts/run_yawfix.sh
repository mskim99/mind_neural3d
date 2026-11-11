#!/bin/bash
# GT 없이 고른 방향(yawsel)으로 Ours GLB를 돌려 공정평가 8지표 다시 (11-11, 사용자, Pro6000 GPU 1)
#  각도 = run_yawsel.sh의 g0 결과(예측 토큰 ↔ 12방위 렌더 DINO) — GT 안 씀 → 정량에 써도 됨
#    g0  : vshape_g0 sub-0001 GLB (TRELLIS 이미지판)                → yawfix_g0
#    txt : 같은 예측 토큰의 텍스트판 "a {예측 범주}" GLB (형상 단계 · seed 같음 → 같은 각도) → yawfix_txt
#  돌린 GLB에 yaw_check(GT 점구름, 진단용)로 어긋남이 줄었는지도 찍음
#  → $FE/out/yawfix_<이름>/values.txt (YAWFIX_DONE). run_yawsel.sh가 아직 돌고 있으면 yawsel.json이 생길 때까지 기다림
# 사용: nohup bash run_yawfix.sh > $W/logs/prism_yawfix.log 2>&1 < /dev/null & disown      (SETS="g0"처럼 고를 수 있음)
source $(dirname $0)/common.sh || exit 1
SETS=${SETS:-"g0 txt"}
FE=$W/fair_eval; PY=$W/venv/bin/python; O=$W/prism_sep/yawsel; SEL=$O/g0/yawsel.json
declare -A GLB=([g0]=$W/prism_vshape/infer/vshape_g0/sub-0001_keep/glb [txt]=$W/prism_vshape/text/glb/vt_s1)
declare -A BASE=([g0]=vshape_g0 [txt]=cap_txt_base)
export FE_PY=$PY FE_WORKERS=${FE_WORKERS:-6} TORCH_HOME=$W/cache/torch FE_BLENDER=${FE_BLENDER:-$W/tools/blender-4.2.3-linux-x64/blender}
V='^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^FPD\s|^P2\|(CD|EMD)\s'
until [ -f $SEL ]; do echo "기다림 $SEL $(date +%H:%M)"; sleep 120; done
grep -h "YAWCMP|g0" $W/logs/prism_yawsel.log 2>/dev/null | head -2
cd $W/run
for s in $SETS; do
  [ $(ls ${GLB[$s]}/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || { echo "GLB 104개 미만 $s ${GLB[$s]}"; continue; }
  g=$O/glb_fix_$s
  $PY $PR/yaw_apply.py --glb ${GLB[$s]} --sel $SEL --out $g
  $PY $PR/yaw_check.py --pred $g --fmt glb --gt_pc $FE/gt_pc/P2 --list $FE/core_test_list.txt --out $O/yaw_after_$s.json --name "돌린뒤_$s" | grep -E "^YAW\|돌린뒤|FAIL"
  echo "=== 평가 시작 yawfix_$s $(date +%H:%M:%S)"
  [ -f $FE/out/yawfix_$s/values.txt ] || bash $FE/run_temp.sh yawfix_$s $g glb $CUDA_VISIBLE_DEVICES > $W/logs/fe_yawfix_$s.log 2>&1
  echo "EVAL_DONE yawfix_$s $(date +%H:%M:%S)"
  echo "  이전 ${BASE[$s]} | $(grep -E "$V" $FE/out/${BASE[$s]}/values.txt 2>/dev/null | tr -s ' \n' ' ')"
  echo "  돌린 뒤 yawfix_$s | $(grep -E "$V" $FE/out/yawfix_$s/values.txt 2>/dev/null | tr -s ' \n' ' ')"
done
echo "=== YAWFIX_DONE $(date +%H:%M:%S)"
