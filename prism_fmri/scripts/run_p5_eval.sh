#!/bin/bash
# P-5 · P-5b 평가 (11-28, Pro6000 GPU 1, 총괄 16:18: 평가는 Pro6000) — 모델마다 한 줄로 겹쳐 돌림
#  ① 7명(1 · 2 · 3 · 5 · 6 · 7 · 8) test 104: 추론 → "a {예측 범주}" 텍스트판 GLB → 방향 보정 → 8지표 (run_branch_subs.sh, 기준 번들 fair_eval_ref)
#  ② AP · APAC: 기존 Ours AP · APAC과 같은 절차(run_apac_final.sh) — 피험자 1 갈래에 sub-0009(AP, test 104) · sub-0011(APAC, 220) fMRI → GLB → 방향 보정 → 형태 지표(FPD · CD · EMD)
#     ②는 그 모델의 ① GLB가 다 만들어진 뒤 시작 (GPU 메모리: 생성 한 줄 약 18 GB)
#  ③ z_B · w 통계: 추론 components.npz(zb · w_ent · w_max)에 들어감 → 로컬 분석
#  BASE=1: 기존 Ours(prism_full/ausvp)도 같은 절차로 7명 + AP · APAC (Pro6000 6번 br_fullausvp_s6은 이미 있어 건너뜀)
#  기존 Ours만 따로: MODELS="fullausvp=$W/prism_full/ausvp/model_final.pt" PACK=p5_eval_base … > logs/p5_eval_base.log
#  모델 파일이 아직 없으면 생길 때까지 기다림 (학습 끝나기 전에 띄워 둬도 됨)
# 사용: [BASE=1] nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_p5_eval.sh > /home/oem/jhj/MinD-3D++/logs/p5_eval.log 2>&1 < /dev/null & disown
GPU=1 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
GPU=1; export CUDA_VISIBLE_DEVICES=1; PY=$W/venv/bin/python; SC=$PR/scripts; TM=microsoft/TRELLIS-text-base
FE2=$W/fair_eval_ref; FEA=$W/fair_eval_ref_apac; export FE_PY=$PY FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$W/cache/torch
MODELS=${MODELS:-"p5bank=$W/prism_p5/ausvp_bank/model_final.pt p5temp=$W/prism_p5/ausvp_bank_temp/model_final.pt"}
[ "${BASE:-0}" = 1 ] && MODELS="$MODELS fullausvp=$W/prism_full/ausvp/model_final.pt"
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cnt() { ls $1/*/*.glb 2>/dev/null | wc -l; }
for f in $FE2/run_eval.sh $DR/annotations/apac_sub0011_list.txt; do [ -f $f ] || echo "주의: 없음 $f"; done
[ -f $FEA/run_eval.sh ] || echo "주의: APAC 번들 없음 $FEA/run_eval.sh → APAC 평가는 건너뜀 (AP는 함)"
echo "=== P-5 평가 시작 $(date +%H:%M:%S) | 모델 $MODELS | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"
apac() {   # 이름 · 체크포인트: AP · APAC 추론 → GLB → 방향 보정 · 형태 지표
  local n=$1 ck=$2 I=$W/prism_p5/apac_$1; mkdir -p $I; cd $W/run
  for X in AP APAC; do
    local S=$([ $X = AP ] && echo sub-0009 || echo sub-0011) L=$([ $X = AP ] && echo $DR/annotations/core_test_list.txt || echo $DR/annotations/apac_sub0011_list.txt) N=$([ $X = AP ] && echo 104 || echo 220)
    local d=$I/${S}_as01_keep
    [ -f $d/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $ck --out $I --subs sub-0001 --fmri_sub $S --list $L --perceptive keep 2>&1 | grep -aE "범주|Error|Traceback"
    $PY $PR/text_prompts.py --comp $d/components.npz --out $I/prompt_$X.json > /dev/null
    [ $(cnt $d/glb) -ge $N ] || trel gen --pred $d/tokens --slat_text $I/prompt_$X.json --text_model $TM --out $d/glb 2>&1 | grep -aE "GEN_DONE|Error|없음"
    echo "GLB|${n}_$X|$(cnt $d/glb)/$N|$(date +%H:%M:%S)"
  done
  SETS="${n}_AP:$I/sub-0009_as01_keep/glb:$I/sub-0009_as01_keep/tokens:shape:v2 ${n}_APAC220:$I/sub-0011_as01_keep/glb:$I/sub-0011_as01_keep/tokens:shape:apac" \
    PAR=2 FE2=$FE2 FEA=$FEA FE_RUN=run_eval.sh GPU=1 bash $SC/run_yaw_v2.sh > $W/logs/p5ev_apac_$n.log 2>&1
  grep -aE "^YV2\|${n}_AP" $W/logs/p5ev_apac_$n.log | grep -v 좌표축
}
one() {   # 이름 · 체크포인트
  local n=$1 ck=$2
  until [ -f $ck ]; do sleep 60; done; sleep 30
  echo "=== $n 시작 $(date +%H:%M:%S) ($ck)"
  NAME=$n CK=$ck SS="1 2 3 5 6 7 8" GENPAR=1 PAR=3 FE2=$FE2 FE_RUN=run_eval.sh GPU=1 bash $SC/run_branch_subs.sh > $W/logs/p5ev_$n.log 2>&1 &
  local B=$!
  until grep -q "=== GLB 끝" $W/logs/p5ev_$n.log 2>/dev/null || [ ! -d /proc/$B ]; do sleep 60; done
  echo "GLB7|$n|7명 GLB 끝|$(date +%H:%M:%S)"
  apac $n $ck
  wait $B; grep -aE "^YV2\|br_${n}_s[0-9] \|" $W/logs/yaw_br_subs_${n}_g1.log
  echo "MODEL_DONE|$n|$(date +%H:%M:%S)"
}
J=""; for kv in $MODELS; do one ${kv%%=*} ${kv#*=} & J="$J $!"; sleep 20; done
wait $J
# 로컬로 가져갈 묶음: 지표 · components(z_B · w) · 학습 로그 · 정성용 정면 렌더(피험자 1 · 6)
cd $W; PK=${PACK:-p5_eval}; P=prism_p5/${PK}_pack.tar; F=""
for kv in $MODELS; do n=${kv%%=*}
  F="$F $(ls -d fair_eval_ref/out/yaw_br_${n}_s*/values.txt fair_eval_ref/out/yaw_${n}_AP/values.txt fair_eval_ref_apac/out/yaw_${n}_APAC220/values.txt prism_branch/subs/$n/sub-000*_keep/components.npz prism_branch/subs/$n/sub-000*_keep/prompt_text.json prism_p5/apac_$n/*/components.npz 2>/dev/null)"
done
F="$F $(ls -d prism_p5/ausvp_bank*/train_log.csv 2>/dev/null)"
tar cf $P $F; tar cf prism_p5/${PK}_render0.tar $(ls fair_eval_ref/out/yaw_br_*_s[16]/render_R2/*/0.png 2>/dev/null | grep -E "br_(p5bank|p5temp|fullausvp)_s[16]/")
echo "=== P5_EVAL_DONE $(date +%H:%M:%S) | $(ls -la $W/prism_p5/${PK}_pack.tar $W/prism_p5/${PK}_render0.tar | awk '{print $5" "$9}' | tr '\n' ' ') | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
