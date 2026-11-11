#!/bin/bash
# 공정평가 기준 v2로 Ours 다시 재기 (11-11, 4090 인계 `_handoff/261111_실험4090→실험Pro6000_공정평가_기준v2적용.md`) — 번들 계산 코드는 그대로 (run_v2.sh)
#  v2 = R2 렌더(그대로) + 기법마다 좌표축 확인 → 예측을 GT 축으로 + PCA 정렬 없는 CD · EMD + 축 맞춘 FPD. v1의 CD · EMD · FPD는 폐기
#  준비: fair_eval_v2_patch.tar를 $W에 풀어 $W/fair_eval_v2/ (스크립트만) → 여기서 v1 번들의 GT 렌더 · 점구름 · FPD 가중치를 링크
#  SETS = "이름:예측 폴더:형식[:v1 이름]" 여러 개. v1 이름을 주고 v1의 tex_R2.json이 있으면 렌더 · 텍스처를 건너뜀(FE_SKIP_TEX=1, 텍스처 5지표는 v1 값 그대로)
#         없으면 렌더부터 전부. 예측 폴더가 없거나 104개 미만이면 건너뜀. PAR개씩 동시에 (CPU 위주, 렌더만 GPU)
#  기본 SETS = 확인값(MinD-3D++ 공개 가중치 추론) + 이미 v1로 잰 Ours 기준 행들
#  → $W/fair_eval_v2/out/<이름>/values.txt (frame · 8지표), 묶음 $W/fair_eval_v2/v2_ours.tar (V2_OURS_DONE)
# 사용: [SETS="…"] [PAR=3] nohup bash run_v2_ours.sh > $W/logs/v2_ours.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
FE1=$W/fair_eval; FE2=$W/fair_eval_v2; PAR=${PAR:-3}; G=$CUDA_VISIBLE_DEVICES
export FE_PY=$W/venv/bin/python FE_BLENDER=${FE_BLENDER:-$W/tools/blender-4.2.3-linux-x64/blender} FE_WORKERS=${FE_WORKERS:-8} FE_THREADS=${FE_THREADS:-8} TORCH_HOME=$W/cache/torch
V=$W/prism_vshape; Y=$W/prism_sep/yawsel
SETS=${SETS:-"mpp_check:$W/spp_eval/verify/pps_meshes:objZ:mpp_check cap_txt_base:$V/text/glb/vt_s1:glb:cap_txt_base vshape_g0:$V/infer/vshape_g0/sub-0001_keep/glb:glb:vshape_g0 yawfix_txt:$Y/glb_fix_txt:glb:yawfix_txt yawfix_g0:$Y/glb_fix_g0:glb:yawfix_g0"}
[ -f $FE2/run_v2.sh ] && [ -f $FE2/scripts/diag_frame.py ] || { echo "없음: $FE2/run_v2.sh (fair_eval_v2_patch.tar를 $W에 풀기)"; exit 1; }
[ -d $FE1/gt_render/R2 ] && [ -d $FE1/gt_pc/P2 ] && [ -d $FE1/fpd ] || { echo "v1 번들 없음: $FE1"; exit 1; }
mkdir -p $FE2/gt_render $FE2/gt_pc_y $FE2/out
[ -e $FE2/gt_render/R2 ] || ln -sfn $FE1/gt_render/R2 $FE2/gt_render/R2
[ -e $FE2/gt_pc_y/P2 ] || ln -sfn $FE1/gt_pc/P2 $FE2/gt_pc_y/P2          # 같은 배열 (README_v2)
[ -e $FE2/fpd ] || ln -sfn $FE1/fpd $FE2/fpd
[ -f $FE2/core_test_list.txt ] || cp $FE1/core_test_list.txt $FE2/
cd $W/run
one() {   # 이름 · 폴더 · 형식 · v1 이름
  local n=$1 d=$2 f=$3 v1=$4 c
  if [ $f = glb ]; then c=$(ls $d/*/*.glb 2>/dev/null | wc -l); else c=$(ls $d/*.obj 2>/dev/null | wc -l); fi
  [ $c -ge 104 ] || { echo "건너뜀 $n (예측 $c개, $d)"; return; }
  if [ ! -f $FE2/out/$n/values.txt ]; then
    mkdir -p $FE2/out/$n; local skip=0
    if [ -n "$v1" ] && [ -f $FE1/out/$v1/tex_R2.json ]; then cp $FE1/out/$v1/tex_R2.json $FE2/out/$n/; skip=1; fi
    echo "=== 시작 $n (렌더 $([ $skip = 1 ] && echo 건너뜀 || echo 함)) $(date +%H:%M:%S)"
    FE_SKIP_TEX=$skip bash $FE2/run_v2.sh $n $d $f $G > $W/logs/v2_$n.log 2>&1
  fi
  echo "V2|$n | $(grep -E '^frame|^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $FE2/out/$n/values.txt 2>/dev/null | tr -s ' \t\n' ' ') $(date +%H:%M:%S)"
  echo "V2|$n 좌표축 분포 | $(grep -A6 '분포' $FE2/out/$n/logs/frame.log 2>/dev/null | tr -s ' \n' ' ' | cut -c1-400)"
  grep -h "주의" $W/logs/v2_$n.log 2>/dev/null
}
for s in $SETS; do
  IFS=: read n d f v1 <<< "$s"
  while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 20; done
  one $n $d $f $v1 &
done
wait
cd $FE2 && tar cf v2_ours.tar $(ls -d out/*/values.txt out/*/frame.txt out/*/logs/frame.log 2>/dev/null) && echo "PACK $(tar tf v2_ours.tar | wc -l)개"
echo "=== V2_OURS_DONE $(date +%H:%M:%S)"
