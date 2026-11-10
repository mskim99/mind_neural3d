#!/bin/bash
# 공정평가 잠정 기준(R2 렌더 + P2 점구름 + FPD, 4090 번들 fair_eval_temp) 을 Pro6000에서 PRISM(Ours)에 적용 (11-10, 인계 261110_실험4090→실험Pro6000)
#  번들 계산 코드는 그대로 (run_temp.sh). 이 스크립트는 환경 · 순서만 정함
#  0 확인: MinD-3D++ 공개 가중치 추론(spp_eval/verify/pps_meshes, objZ) → 4090 확인값과 대조
#  1 Ours: 두 줄로 나눠 차례로 (GLB 104개가 다 생길 때까지 기다렸다가 시작)
#  → $FE/out/<이름>/values.txt, 묶음 $FE/fair_ours.tar (FAIR_OURS_DONE)
# 사용: nohup bash run_fair_ours.sh > $W/logs/fair_ours.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
FE=$W/fair_eval; G=$CUDA_VISIBLE_DEVICES
export FE_PY=$W/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender FE_WORKERS=${FE_WORKERS:-6} FE_THREADS=${FE_THREADS:-6}
export TORCH_HOME=$W/cache/torch                          # ViT-H-14 가중치 (~/.cache 금지)
mkdir -p $TORCH_HOME
[ -f $FE/run_temp.sh ] || { echo "번들 없음 $FE"; exit 1; }
$FE_PY -c "import torch, torchvision, lpips, ot, trimesh, scipy, skimage, pandas, PIL; print('패키지 OK trimesh', trimesh.__version__)" || exit 1

declare -A P=(
  [mpp_check]=$W/spp_eval/verify/pps_meshes
  [text_s1]=$W/prism_text/glb/text_s1
  [vt_s1]=$W/prism_vshape/text/glb/vt_s1
  [vshape_g0]=$W/prism_vshape/infer/vshape_g0/sub-0001_keep/glb
  [fig]=$W/prism_infer_fig/sub-0001_keep/glb
  [text_s8]=$W/prism_text/glb/text_s8
  [vt_s8]=$W/prism_vshape/text/glb/vt_s8
  [fig_s8]=$W/prism_sub8/glb/fig_s8
  [vshape_g1]=$W/prism_vshape/infer/vshape_g1/sub-0001_keep/glb
)
cnt() { if [ $1 = mpp_check ]; then ls ${P[$1]}/*.obj 2>/dev/null | wc -l; else ls ${P[$1]}/*/*.glb 2>/dev/null | wc -l; fi; }
one() {   # 이름
  local n=$1 f=glb; [ $n = mpp_check ] && f=objZ
  [ -f $FE/out/$n/values.txt ] && { echo "이미 있음 $n"; return; }
  until [ $(cnt $n) -ge 104 ]; do echo "기다림 $n ($(cnt $n)/104) $(date +%H:%M)"; sleep 600; done
  echo "=== 시작 $n $(date +%H:%M:%S)"
  bash $FE/run_temp.sh $n ${P[$n]} $f $G > $W/logs/fair_$n.log 2>&1
  echo "=== 끝 $n $(date +%H:%M:%S) | $(grep -cE . $FE/out/$n/values.txt 2>/dev/null)줄 | $(grep -E '예측 파일 없음|TEMP_DONE' $W/logs/fair_$n.log | tr '\n' ' ')"
}
cd $FE
( for n in mpp_check text_s1 vt_s1 text_s8 vt_s8; do one $n; done ) & A=$!
sleep 60
( for n in vshape_g0 fig fig_s8 vshape_g1; do one $n; done ) & B=$!
wait $A $B

echo "=== 결과"
for n in mpp_check text_s1 vt_s1 vshape_g0 fig text_s8 vt_s8 fig_s8 vshape_g1; do
  echo "--- $n"; grep -E "^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^P2\|(CD|EMD)\s|^FPD\s" $FE/out/$n/values.txt 2>/dev/null | tr '\n' ' '; echo
done
tar cf $FE/fair_ours.tar -C $FE out/*/values.txt out/*/fpd.json out/*/tex_R2.json 2>/dev/null
echo "PACK $(tar tf $FE/fair_ours.tar | wc -l)개"
echo "=== FAIR_OURS_DONE $(date +%H:%M:%S)"
