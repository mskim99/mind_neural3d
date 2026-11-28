#!/bin/bash
# V-10 피험자 6번 (11-28, Pro6000 GPU 1) — run_v10.sh(5090, 피험자 1번)와 같은 실험을 Pro6000 경로로. 비교 줄은 모두 이 서버에서 다시 만듦
#  ① 기존 Ours(7명 모델 prism_full/ausvp, 조향 없음 "a {범주}") 피험자 6번: 추론 → 텍스트판 GLB → 방향 보정 → 8지표 (run_branch_subs.sh, 세트 br_fullausvp_s6)
#  ② Appearance Head 학습(v10_head.py, 기존 Ours 고정) → steer_s06.json (α = 0.5 × clip(ŝ_PC1, ±2))
#  ③ 정답 값 조향(상한) zgt_s06.json (v8_steer_json, 정답 z = 자극 band_hi)
#  ④ 두 조향 세트 생성(형상 · seed · 기준 문장 = ①과 같음) → ①의 yawsel로 회전 → 진단 렌더 → 반응 · 질감 표 → 8지표
# 필요: $W/prism_v9/{v9_feat.npz, v5_feat.npz} (로컬에서 scp, 5090과 md5 같은 파일)
# 사용: nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_v10_p6.sh > /home/oem/jhj/MinD-3D++/logs/v10_s06.log 2>&1 < /dev/null & disown
GPU=1 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
GPU=1; export CUDA_VISIBLE_DEVICES=1; PY=$W/venv/bin/python; SC=$PR/scripts; SN=s06; SUB=sub-0006
CK=$W/prism_full/ausvp/model_final.pt; H=$W/prism_v10; O=$H/s06; mkdir -p $O/react $O/diag
B=$W/prism_branch/subs/fullausvp/${SUB}_keep; TOK=$B/tokens; P0=$B/prompt_text.json; SEL=$W/prism_yaw/br_fullausvp_s6/yawsel.json
FE2=$W/fair_eval_ref; export FE_PY=$PY FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender FE_WORKERS=8 FE_THREADS=8 TORCH_HOME=$W/cache/torch
TM=microsoft/TRELLIS-text-base
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
diag() { (cd $MG && source $MG/mg_guard.sh >/dev/null && python $PR/v8_diag_render.py --glb $1 --out $2) 2>&1 | grep -E "DONE|Error"; }
freewait() { until [ "$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 30; done; }
cnt() { ls $1/*/*.glb 2>/dev/null | wc -l; }
for f in $CK $W/prism_v9/v9_feat.npz $W/prism_v9/v5_feat.npz $FE2/run_eval.sh; do [ -f $f ] || { echo "없음 $f"; exit 1; }; done
echo "=== V-10 s06 시작 $(date +%H:%M:%S) | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"
# ① 기존 Ours s06 (GLB + 방향 보정 + 8지표). 8지표는 뒤에서 돌게 두고, yawsel이 나오면 다음으로
wait_gpu_free base_s06
NAME=fullausvp CK=$CK SS=6 GENPAR=1 PAR=2 FE2=$FE2 FE_RUN=run_eval.sh GPU=1 bash $SC/run_branch_subs.sh > $W/logs/v10_s06_base.log 2>&1 & BASE=$!
until [ -f $SEL ] && [ $(cnt $W/prism_yaw/br_fullausvp_s6/glb) -ge 104 ]; do sleep 60; [ -d /proc/$BASE ] || { [ -f $SEL ] && break; echo "기존 Ours s06 실패 (logs/v10_s06_base.log)"; exit 1; }; done
echo "STAGE|base GLB $(cnt $B/glb_text) · 방향 보정 끝|$(date +%H:%M:%S)"
# ② head 학습 (GPU 메모리 작음)
wait_gpu_free v10_head_s06
[ -f $H/steer_$SN.json ] || (cd $W/run && $PY $PR/v10_head.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --v9 $W/prism_v9/v9_feat.npz --v5 $W/prism_v9/v5_feat.npz \
    --sub $SUB --base_prompt $P0 --out $H) | grep -vE "^\s*$"
[ -f $H/steer_$SN.json ] || { echo "head 실패 $SN"; exit 1; }
echo "STAGE|head 끝 $($PY -c "import json;r=json.load(open('$H/head_$SN.json'));print(r['chosen'],r['test'])")|$(date +%H:%M:%S)"
# ③ 정답 값 조향 (상한)
$PY $PR/v8_steer_json.py --base $P0 --v5 $W/prism_v9/v5_feat.npz --ann $DR/annotations --mode z --src gt --k 0.5 \
  --plus '{an} with fine surface details and patterns' --minus '{an} with a smooth plain surface' --out $H/zgt_$SN.json
# ④ 조향 생성 → 회전 → 진단 렌더
[ -d $O/diag/base_$SN ] && [ $(ls $O/diag/base_$SN | wc -l) -ge 104 ] || diag $W/prism_yaw/br_fullausvp_s6/glb $O/diag/base_$SN
for kv in ah_$SN=$H/steer_$SN.json zgt_$SN=$H/zgt_$SN.json; do
  n=${kv%%=*}; j=${kv#*=}; mkdir -p $O/$n
  wait_gpu_free v10_$n; freewait 19000
  echo "=== $n 생성 시작 $(date +%H:%M:%S)"
  [ $(cnt $O/$n/glb) -ge 104 ] || trel gen --pred $TOK --slat_text $P0 --slat_steer $j --text_model $TM --out $O/$n/glb | grep -E "GEN_DONE|Error|조향|없음"
  $PY $PR/yaw_apply.py --glb $O/$n/glb --sel $SEL --out $O/$n/yawg | grep -o "YAWAPPLY_DONE [0-9]*개"
  diag $O/$n/yawg $O/diag/$n
  echo "STEER|$n|GLB $(cnt $O/$n/glb)|$(date +%H:%M:%S)"
done
$PY -I $PR/v8_react_feat.py --diag $O/diag --sets base_$SN,ah_$SN,zgt_$SN --out $O/react | grep REACT
$PY -I $PR/v10_render_tex.py --diag $O/diag --sets base_$SN,ah_$SN,zgt_$SN --out $O/react | grep TEX
echo "DIAG_READY|$SN|$(date +%H:%M:%S)"
# 8지표 (기존 Ours가 끝난 뒤 하나씩)
wait $BASE; echo "CV|base_$SN | $(grep -E '^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $FE2/out/yaw_br_fullausvp_s6/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(date +%H:%M:%S)"
for n in ah_$SN zgt_$SN; do
  E=yaw_v10_$n; wait_gpu_free eval_$n; freewait 8000
  [ -f $FE2/out/$E/values.txt ] || bash $FE2/run_eval.sh $E $O/$n/yawg glb 1 > $W/logs/eval_$E.log 2>&1
  echo "CV|$n | $(grep -E '^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $FE2/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(date +%H:%M:%S)"
done
cd $W && tar cf $H/v10_s06_pack.tar prism_v10/head_$SN.json prism_v10/head_$SN.npz prism_v10/steer_$SN.json prism_v10/zgt_$SN.json prism_v10/s06/react \
  prism_branch/subs/fullausvp/${SUB}_keep/prompt_text.json fair_eval_ref/out/yaw_br_fullausvp_s6/values.txt fair_eval_ref/out/yaw_v10_ah_$SN/values.txt fair_eval_ref/out/yaw_v10_zgt_$SN/values.txt 2>/dev/null
tar cf $H/v10_s06_render.tar -C $W prism_v10/s06/diag $(cd $W && ls -d fair_eval_ref/out/yaw_br_fullausvp_s6/render_R2 fair_eval_ref/out/yaw_v10_*_$SN/render_R2 2>/dev/null)
echo "=== V10_S06_DONE $(date +%H:%M:%S) | $(ls -la $H/v10_s06_pack.tar $H/v10_s06_render.tar | awk '{print $5" "$9}' | tr '\n' ' ')"
