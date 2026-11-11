#!/bin/bash
# AP · APAC를 공정평가 기준 v2로 (11-11) — 형태 지표만(FPD · CD · EMD, 렌더 없음). 번들 계산 코드는 그대로 (run_v2.sh, FE_SKIP_TEX=1)
#  예측은 11-10 run_apac.sh가 만든 것 그대로: Ours = Subject 1 갈래에 sub-0009(AP) · sub-0011(APAC) fMRI, MinD-3D++ = 공개 가중치 추론 (있을 때만)
#  AP   = core test 104 목록 그대로 → $W/fair_eval_v2 에서 실행
#  APAC = sub-0011 실제 220 목록 → 옆 폴더 $W/fair_eval_v2_apac (run_v2.sh 복사 + scripts · fpd 링크, 목록만 apac_sub0011_list.txt, GT = v1 gt_pc/P2에 풀어 둔 APAC 점구름)
#         run_v2.sh가 찍는 "예측 파일 없음 n / 104"의 104는 고정 문구 (실제 물체 수는 values.txt의 dist|n_objects)
#  → out/{ours,pps}_AP/values.txt ($W/fair_eval_v2), out/{ours,pps}_APAC220/values.txt ($W/fair_eval_v2_apac) (V2_APAC_DONE)
# 사용: nohup bash run_v2_apac.sh > $W/logs/v2_apac.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
FE1=$W/fair_eval; FE2=$W/fair_eval_v2; FEA=$W/fair_eval_v2_apac; L=$DR/annotations/apac_sub0011_list.txt; I=$W/prism_apac
export FE_PY=$W/venv/bin/python FE_WORKERS=${FE_WORKERS:-8} FE_THREADS=${FE_THREADS:-8} TORCH_HOME=$W/cache/torch FE_SKIP_TEX=1
[ -f $FE2/run_v2.sh ] && [ -e $FE2/gt_pc_y/P2 ] || { echo "v2 번들 없음 (run_v2_ours.sh 먼저)"; exit 1; }
[ -f $L ] || { echo "목록 없음 $L"; exit 1; }
M=$(while IFS=/ read s u; do u=${u%$'\r'}; [ -f $FE1/gt_pc/P2/$u.npy ] || echo x; done < $L | wc -l); [ $M -eq 0 ] || { echo "APAC GT 점구름 $M개 없음 ($FE1/gt_pc/P2)"; exit 1; }
mkdir -p $FEA/gt_pc_y $FEA/out; cp $FE2/run_v2.sh $FEA/; cp $L $FEA/core_test_list.txt
[ -e $FEA/scripts ] || ln -sfn $FE2/scripts $FEA/scripts; [ -e $FEA/fpd ] || ln -sfn $FE1/fpd $FEA/fpd; [ -e $FEA/gt_pc_y/P2 ] || ln -sfn $FE1/gt_pc/P2 $FEA/gt_pc_y/P2
cd $W/run
one() {   # 번들 폴더 · 이름 · 예측 폴더 · 형식 · 필요한 개수
  local R=$1 n=$2 d=$3 f=$4 need=$5 c
  if [ $f = glb ]; then c=$(ls $d/*/*.glb 2>/dev/null | wc -l); else c=$(ls $d/*.obj 2>/dev/null | wc -l); fi
  [ $c -ge $need ] || { echo "건너뜀 $n (예측 $c/$need, $d)"; return; }
  [ -f $R/out/$n/values.txt ] || { echo "=== 시작 $n $(date +%H:%M:%S)"; bash $R/run_v2.sh $n $d $f $CUDA_VISIBLE_DEVICES > $W/logs/v2_$n.log 2>&1; }
  echo "V2|$n | $(grep -E '^frame|^(CD|EMD|FPD)\s|n_objects|FPD\|n' $R/out/$n/values.txt 2>/dev/null | tr -s ' \t\n' ' ') $(date +%H:%M:%S)"
  echo "V2|$n 좌표축 분포 | $(grep '분포' $R/out/$n/logs/frame.log 2>/dev/null | cut -c1-300)"
}
one $FE2 ours_AP $I/ours/sub-0009_as01_keep/glb glb 104 &
one $FEA ours_APAC220 $I/ours/sub-0011_as01_keep/glb glb 220 &
one $FE2 pps_AP $I/pps_AP/meshes objZ 104 &
wait
one $FEA pps_APAC220 $I/pps_APAC/meshes objZ 220
echo "=== V2_APAC_DONE $(date +%H:%M:%S)"
