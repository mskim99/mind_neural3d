#!/bin/bash
# vshape_g0(현재 버전) 피험자 추가 공정평가 (11-10, 사용자: 잘 나오던 피험자 2명 더): 추론 → GLB → 잠정 기준(R2 · P2 · FPD)
#  기본 sub-0003 · sub-0002 (test 범주 정확도 0.423 · 0.337, sub-0001 다음으로 높음). GLB는 차례로, 평가는 GLB 끝난 피험자부터 뒤에서
#  GLB 경로는 run_vshape_allsub.sh와 같음 (이미 있으면 건너뜀)
#  → $FE/out/vshape_g0_<sub>/values.txt, 묶음 $FE/fair_subs.tar (FAIR_SUBS_DONE)
# 사용: [SUBS="sub-0003 sub-0002"] nohup bash run_fair_subs.sh > $W/logs/fair_subs.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
SUBS=${SUBS:-"sub-0003 sub-0002"}
FE=$W/fair_eval; G=$CUDA_VISIBLE_DEVICES; PY=$W/venv/bin/python
CK=$W/prism_vshape/vshape_g0/model_final.pt; I=$W/prism_vshape/infer/vshape_g0
export FE_PY=$PY FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender FE_WORKERS=${FE_WORKERS:-6} FE_THREADS=${FE_THREADS:-6} TORCH_HOME=$W/cache/torch
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
[ -f $FE/run_temp.sh ] && [ -f $CK ] || { echo "없음: 번들 또는 $CK"; exit 1; }
cd $W/run
ev() {   # 피험자
  local n=vshape_g0_$1
  [ -f $FE/out/$n/values.txt ] && { echo "이미 있음 $n"; return; }
  echo "=== 평가 시작 $n $(date +%H:%M:%S)"
  bash $FE/run_temp.sh $n $I/$1_keep/glb glb $G > $W/logs/fair_$n.log 2>&1
  echo "=== 평가 끝 $n $(date +%H:%M:%S) | $(grep -E '예측 파일 없음' $W/logs/fair_$n.log)"
}
EJ=""
for s in $SUBS; do
  [ -f $I/${s}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I --subs $s --perceptive keep | grep -E "범주|DONE|Error"
  [ $(ls $I/${s}_keep/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/${s}_keep/tokens --out $I/${s}_keep/glb | grep -E "GEN_DONE|Error"
  echo "GLB $s $(ls $I/${s}_keep/glb/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ $(ls $I/${s}_keep/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || { echo "GLB $s 104개 미만 → 멈춤"; exit 1; }
  ev $s & EJ="$EJ $!"
done
wait $EJ
echo "=== 결과"
for s in $SUBS; do echo "--- $s"; grep -E "^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^P2\|(CD|EMD)\s|^FPD\s" $FE/out/vshape_g0_$s/values.txt 2>/dev/null | tr '\n' ' '; echo; done
cd $FE && tar cf fair_subs.tar $(for s in $SUBS; do echo out/vshape_g0_$s/values.txt; done) 2>/dev/null; echo "PACK $(tar tf fair_subs.tar | wc -l)개"
echo "=== FAIR_SUBS_DONE $(date +%H:%M:%S)"
