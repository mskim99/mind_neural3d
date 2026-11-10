#!/bin/bash
# AP · APAC (11-10, MinD-3D++ TPAMI Tab.3: "model trained on Subject 1", AP = Subject 9 · core test 104, APAC = Subject 11 · 220물체 55범주) — FPD · CD · EMD만
#  Ours: vshape_g0의 Subject 1 갈래에 sub-0009 · sub-0011 fMRI를 넣어 추론 (infer_prism.py --fmri_sub) → GLB → fe_shape
#  MinD-3D++: PPS_CKPT(공개 가중치 Shape sub01 전체 체크포인트)가 있으면 같은 fMRI로 공식 절차 추론(LoRA 미적용, infer_mind3dpp_batch.py) → OBJ → fe_shape
#  준비(로컬에서 올림): $DR/sub-0009 · $DR/sub-0011 (npy_data), $DR/annotations/apac_list.txt, APAC GT 점구름 apac_gt_p2.tar → $FE/gt_pc/P2/ 에 풂
#  → $FE/out/{ours,pps}_{AP,APAC}/values.txt, 묶음 $FE/apac.tar (APAC_DONE)
# 사용: [PPS_CKPT=<경로>] [PPS_ONLY=1 (MinD-3D++만, 예: GPU=0)] nohup bash run_apac.sh > $W/logs/apac.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
source $(dirname $0)/fe_shape.sh
FE=${FE:-$W/fair_eval}; PY=$W/venv/bin/python; CK=$W/prism_vshape/vshape_g0/model_final.pt; I=$W/prism_apac/ours
export FE_PY=$PY FE_WORKERS=${FE_WORKERS:-6}
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
declare -A FS=([AP]=sub-0009 [APAC]=sub-0011) LS=([AP]=$DR/annotations/core_test_list.txt [APAC]=$DR/annotations/apac_list.txt) NN=([AP]=104 [APAC]=220)
for x in AP APAC; do
  [ $(find $DR/${FS[$x]}/npy_data -name '*.npy' 2>/dev/null | wc -l) -ge ${NN[$x]} ] || { echo "fMRI 없음: $DR/${FS[$x]}"; exit 1; }
  [ -f ${LS[$x]} ] || { echo "목록 없음 ${LS[$x]}"; exit 1; }
done
[ $(ls $FE/gt_pc/P2 | wc -l) -ge 300 ] || { echo "APAC GT 점구름 없음 (apac_gt_p2.tar를 $FE/gt_pc/P2에 풀기)"; exit 1; }
cd $W/run; mkdir -p $I

pps() {   # MinD-3D++ (뒤에서)
  [ -f "$PPS_CKPT" ] || { echo "MinD-3D++ 건너뜀 (PPS_CKPT 없음)"; return; }
  for x in AP APAC; do
    local o=$W/prism_apac/pps_$x
    if [ $(ls $o/meshes/*.obj 2>/dev/null | wc -l) -lt ${NN[$x]} ]; then
      echo "=== MinD-3D++ 추론 $x $(date +%H:%M:%S)"
      (cd $W/MinD-3D/InstantMesh && $PY $W/scripts/infer_mind3dpp_batch.py ../configs/mind3d_pp_infer.yaml --dataset shape --list ${LS[$x]} --fmri_dir $DR/${FS[$x]} \
        --unet_path $PPS_CKPT --z123 $W/weights/zero123plus-v1.2 --lrm_ckpt $W/weights/InstantMesh/instant_mesh_large.ckpt --out_dir $o > $W/logs/apac_pps_$x.log 2>&1)
    fi
    echo "OBJ pps_$x $(ls $o/meshes/*.obj 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
    fe_shape pps_$x $o/meshes obj ${LS[$x]}
  done
}
pps & PJ=$!
[ -n "$PPS_ONLY" ] && { wait $PJ; echo "=== APAC_DONE (MinD-3D++만) $(date +%H:%M:%S)"; exit 0; }

for x in AP APAC; do
  d=$I/${FS[$x]}_as01_keep
  [ -f $d/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I --subs sub-0001 --fmri_sub ${FS[$x]} --list ${LS[$x]} --perceptive keep | grep -E "범주|DONE|Error"
  [ $(ls $d/glb/*/*.glb 2>/dev/null | wc -l) -ge ${NN[$x]} ] || trel gen --pred $d/tokens --out $d/glb | grep -E "GEN_DONE|Error"
  echo "GLB ours_$x $(ls $d/glb/*/*.glb 2>/dev/null | wc -l)개 (예상 ${NN[$x]}) $(date +%H:%M:%S)"
  fe_shape ours_$x $d/glb glb ${LS[$x]}
done
wait $PJ
echo "=== 결과"; for n in ours_AP ours_APAC pps_AP pps_APAC; do echo "$n $(grep -E '^P2\|(CD|EMD)\s|^FPD\s|n_objects' $FE/out/$n/values.txt 2>/dev/null | tr '\n' ' ')"; done
cd $FE && tar cf apac.tar $(ls -d out/{ours,pps}_{AP,APAC}/values.txt 2>/dev/null) && echo "PACK $(tar tf apac.tar | wc -l)개"
echo "=== APAC_DONE $(date +%H:%M:%S)"
