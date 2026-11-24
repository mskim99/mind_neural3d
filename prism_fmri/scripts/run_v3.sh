#!/bin/bash
# V-3 (11-24, 5090): (가) 세 축 비율을 fMRI에서 읽기(8명) → (나) 최종 Ours 피험자 1 · 6 메시를 조건 5개로 조정해 형태 지표(CD · EMD · FPD)만 다시 잼 → 정성 9물체 렌더
# 사용 (절대 경로로): nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_v3.sh > $W/logs/v3.log 2>&1 < /dev/null &
GPU=0 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
cd $W/run; PY=$W/venv/bin/python; O=$W/prism_v3; mkdir -p $O
F=/home/oem/jhj/fair_eval; V=$F/fair_eval; export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home FE_SKIP_TEX=1 FE_WORKERS=6
S8=sub-0001,sub-0006,sub-0002,sub-0003,sub-0004,sub-0005,sub-0007,sub-0008
if [ ! -f $O/READ_DONE ]; then wait_gpu_free v3_read; echo "=== v3_read $(date +%H:%M:%S)"
  $PY $PR/v3_read.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_full/ausvp/model_final.pt --vox $W/data/shape_targets/voxels --subs $S8 --out $O && touch $O/READ_DONE; fi
[ -f $O/READ_DONE ] || { echo "v3_read 실패"; exit 1; }
for s in 01 06; do
  D=$O/s$s
  [ -f $D/v3.json ] || $FE_PY $PR/v3_rescale.py --glb $W/prism_yaw/fullausvp_s$s/glb --comp $W/prism_full/eval_ausvp/infer/s$s/sub-00${s}_keep/components.npz --pred $O/pred_sub-00$s.npz --catmean $O/cat_mean.npz \
    --data_root $DR --gt_pc $V/gt_pc/P2 --out $D | cut -c1-1200
  for c in fenc predcat gtcat other fraw; do
    E=v3_${c}_s$s
    [ -f $V/out/$E/values.txt ] || { wait_gpu_free $E; bash $V/run_eval.sh $E $D/$c glb $GPU > $F/logs/eval_$E.log 2>&1; }
    echo "V3|s$s|$c | $(grep -E '^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(date +%H:%M:%S)"
  done
done
U="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944 4dc9f86110ea40aaa570c6c691c987a8 a51410ca51a40c2529412f6cb23454eb 17072ff9ec47bfadc3fd2392bc923b2f 969f830a531682cef155d75bbf62b80"
for s in 01 06; do for c in fenc predcat; do
  D=$O/s$s; for u in $U; do echo $D/$c/$u/$u.glb; done > $D/qual_$c.txt
  [ -d $D/render_$c ] || $FE_BLENDER -b -t 4 -P $V/scripts/render.py -- --list $D/qual_$c.txt --out_root $D/render_$c --res 1024 --light 8000 --camera_dist 1.5 --fov 56 --views 0:30,60:30,120:30,180:30,240:30,300:30 --metallic0 2>&1 | grep -E "FAIL" | head -3
done; done
echo "=== V3_DONE $(date +%H:%M:%S)"
