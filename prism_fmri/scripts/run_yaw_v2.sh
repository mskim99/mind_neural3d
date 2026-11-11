#!/bin/bash
# 방향 보정(GT 없이, yawsel)을 Ours 전 세트에 넣고 공정평가 v2 (11-11 사용자 결정: "전 세트에 적용")
#  근거: Subject 1 기준 행 CD 1.879 → 1.345 · FPD 1.698 → 1.264 (`.보고 자료/261112_9차보고/공정평가/Ours_기준v2.md`)
#  세트마다: ① 그 세트 GLB를 자극 카메라(거리 1.5 · FOV 56 · 고도 30)로 12방위 렌더(512²)
#            ② trellis_tools yawsel: 렌더 DINO 토큰이 그 세트의 예측 토큰(자극 프레임 0 흉내)과 가장 비슷한 방위 → 되돌릴 회전 (GT 안 씀)
#            ③ yaw_apply: GLB를 그만큼 돌림 → $W/prism_yaw/<이름>/glb
#            ④ v2 평가 (번들 run_v2.sh 그대로): full = 8지표(렌더 포함) / shape = 형태 지표만(FE_SKIP_TEX=1, AP · APAC)
#  SETS = "이름:GLB 폴더:예측 토큰 폴더:full|shape:v2|apac" 여러 개 (apac = 목록 220짜리 옆 번들 $FEA, run_v2_apac.sh가 만든 것)
#  PAR개씩 동시에, WAIT=1이면 GLB가 NEED(기본 104, apac 220)개가 될 때까지 기다림. vt_s1은 GT 진단(yaw_check)과 비교한 YAWCMP 줄도 찍음
#  → $FE2/out/yaw_<이름>/values.txt (apac은 $FEA/out), 로그 YV2|… (YAW_V2_DONE)
#  PRESET=pro6000 (ablation w/o all · w/o P · only P + 피험자 1 · 8 · 2 · 3 · 5 · 6 · 7 + AP · APAC) / PRESET=5090 (only S · only V · w/o S · w/o V)
# 사용: PRESET=pro6000 | SETS="…" [PAR=4] [WAIT=1] [FE2=…] nohup bash run_yaw_v2.sh > $W/logs/yaw_v2.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
A=$W/prism_abl/infer; T=$W/prism_vshape/text/glb; I=$W/prism_vshape/infer/vshape_g0; K=sub-0001_keep; AP=$W/prism_apac/ours
ab() { echo "abl_$1:$A/$1/$K/glb_text:$A/$1/$K/tokens:full:v2"; }; sb() { echo "vt_s$1:$T/vt_s$1:$I/sub-000$1_keep/tokens:full:v2"; }
case "${PRESET:-}" in   # 미리 정한 묶음 (ablation 먼저)
  pro6000) SETS="$(ab noSVP) $(ab noP) $(ab onlyP) $(sb 1) $(sb 8) $(sb 2) $(sb 3) $(sb 5) $(sb 6) $(sb 7) ours_AP:$AP/sub-0009_as01_keep/glb:$AP/sub-0009_as01_keep/tokens:shape:v2 ours_APAC220:$AP/sub-0011_as01_keep/glb:$AP/sub-0011_as01_keep/tokens:shape:apac";;
  5090)    SETS="$(ab onlyS) $(ab onlyV) $(ab noS) $(ab noV)";;
esac
: ${SETS:?SETS="이름:GLB 폴더:토큰 폴더:full|shape:v2|apac …"}
FE2=${FE2:-$W/fair_eval_v2}; FEA=${FEA:-$W/fair_eval_v2_apac}; PAR=${PAR:-4}; G=$CUDA_VISIBLE_DEVICES; Y=$W/prism_yaw; PY=$W/venv/bin/python
export FE_PY=$PY FE_BLENDER=${FE_BLENDER:-$W/tools/blender-4.2.3-linux-x64/blender} FE_WORKERS=${FE_WORKERS:-8} FE_THREADS=${FE_THREADS:-8} TORCH_HOME=$W/cache/torch
VIEWS=$(seq -s, 0 30 330 | sed 's/\([0-9]*\)/\1:30/g')
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
[ -f $FE2/run_v2.sh ] && [ -f $FE2/scripts/render.py ] || { echo "v2 번들 없음 $FE2"; exit 1; }
cd $W/run; mkdir -p $Y
one() {   # 이름 · GLB 폴더 · 토큰 폴더 · full|shape · v2|apac
  local n=$1 d=$2 tk=$3 mode=$4 rt=$5 O=$Y/$1 R=$FE2 need=104
  [ "$rt" = apac ] && { R=$FEA; need=220; }
  cnt() { ls $d/*/*.glb 2>/dev/null | wc -l; }
  if [ $(cnt) -lt $need ] && [ "${WAIT:-0}" = 1 ]; then echo "기다림 $n ($(cnt)/$need) $(date +%H:%M:%S)"; until [ $(cnt) -ge $need ]; do sleep 60; done; sleep 60; fi
  [ $(cnt) -ge $need ] && [ $(ls $tk/*.npy 2>/dev/null | wc -l) -ge $need ] || { echo "건너뜀 $n (GLB $(cnt) · 토큰 $(ls $tk/*.npy 2>/dev/null | wc -l) / $need)"; return; }
  [ -f $R/run_v2.sh ] || { echo "건너뜀 $n (번들 없음 $R — run_v2_apac.sh 먼저)"; return; }
  mkdir -p $O
  if [ ! -f $O/yawsel.json ]; then
    echo "=== $n 12방위 렌더 $(date +%H:%M:%S)"; ls $d/*/*.glb > $O/list.txt
    $FE_BLENDER -b -t 4 -P $FE2/scripts/render.py -- --list $O/list.txt --out_root $O/render --res 512 --light 8000 --camera_dist 1.5 --fov 56 --views $VIEWS --metallic0 2>&1 | grep -E "FAIL" | head -5
    echo "=== $n 방향 고르기 $(date +%H:%M:%S)"
    trel yawsel --frames $O/render --pred $tk --out $O --nview 12 | grep -E "YAWSEL_DONE|Error|없음"
  fi
  [ -f $O/yawsel.json ] || { echo "실패 $n (yawsel.json 없음)"; return; }
  [ $(ls $O/glb/*/*.glb 2>/dev/null | wc -l) -ge $need ] || $PY $PR/yaw_apply.py --glb $d --sel $O/yawsel.json --out $O/glb
  [ $n = vt_s1 ] && [ -f $W/prism_sep/yaw/ours_g0.json ] && $PY $PR/yaw_compare.py --yaw $W/prism_sep/yaw/ours_g0.json --sel $O/yawsel.json --name vt_s1_txt | grep -E "^YAWCMP\|vt_s1_txt (정한|반대)"
  if [ ! -f $R/out/yaw_$n/values.txt ]; then
    echo "=== $n v2 평가 ($mode) $(date +%H:%M:%S)"
    FE_SKIP_TEX=$([ $mode = shape ] && echo 1 || echo 0) bash $R/run_v2.sh yaw_$n $O/glb glb $G > $W/logs/yawv2_$n.log 2>&1
  fi
  echo "YV2|$n | $(grep -E '^frame|^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s|dist\|n_objects' $R/out/yaw_$n/values.txt 2>/dev/null | tr -s ' \t\n' ' ') $(date +%H:%M:%S)"
  echo "YV2|$n 좌표축 분포 | $(grep '분포' $R/out/yaw_$n/logs/frame.log 2>/dev/null | cut -c1-260)"
}
for s in $SETS; do
  IFS=: read n d tk mode rt <<< "$s"
  while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 20; done
  one $n $d $tk ${mode:-full} ${rt:-v2} &
done
wait
echo "=== YAW_V2_DONE $(date +%H:%M:%S)"
