#!/bin/bash
# F-1 (11-28, 5090 GPU 0): 최종 Ours(ausvp) + Statistic Regressor(모델 안, L_app)를 7명 처음부터 학습 → 조향 없음 · 있음 생성 → 진단 렌더 · Blender 8지표
#  ① V-10 끝(V10_DONE) 기다림 ② 정답 app_target.npz ③ w 고르기: 7명 · 1,000 스텝 · 검증 물체(HOLD 8) — w 0(기존) · 1.0 · 0.1, 다른 손실(tok · proto · vt) 합이 기존의 +2 % 안인 가장 큰 w
#  ④ 2만 스텝 7명(HOLD 0, 기존 Ours와 같음) ⑤ 피험자 1번 먼저: 조향 없음 평가(run_sub_eval) → 조향 생성 · 진단 렌더 · Blender ⑥ 나머지 6명 같은 순서
# 사용 (절대 경로로): nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_f1.sh > $W/logs/f1.log 2>&1 < /dev/null &
GPU=0 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
GPU=0; export CUDA_VISIBLE_DEVICES=0; PY=$W/venv/bin/python; F=$W/prism_f1; mkdir -p $F/react
FE=/home/oem/jhj/fair_eval; V=$FE/fair_eval; export FE_PY=$FE/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$FE/torch_home
echo "=== V-10 끝 기다림 $(date +%H:%M:%S)"; until grep -q V10_DONE $W/logs/v10.log; do sleep 60; done
[ -f $TGT/app_target.npz ] || $PY $PR/prep_app_target.py --data_root $DR --v9 $W/prism_v9/v9_feat.npz --v5 $W/prism_v9/v5_feat.npz --out $TGT/app_target.npz
echo "=== w 고르기 (1,000 스텝 · 검증 물체) $(date +%H:%M:%S)"
POLITE=0 MAXJ=2 HOLD=8 OUT=$W/prism_f1tune LP=prism_f1tune STEPS=1000 SAVE="" GPU=0 RUNS="ausvp ausvp_app ausvp_app_w01" bash $PR/scripts/run_branch.sh
RUN=$($PY - <<'PY'
import csv, sys
def other(n):
    r = [x for x in csv.DictReader(open(f'/home/oem/jhj/MinD-3D++/prism_f1tune/{n}/train_log.csv')) if x['split'] == 'hold'][-1]
    return sum(float(r[k]) for k in ('tok_mse', 'tok_cos', 'proto', 'vt')), r.get('app')
b, _ = other('ausvp'); pick = 'ausvp_app_w01'
for n in ('ausvp_app', 'ausvp_app_w01'):
    o, ap = other(n); print(f'{n}: 다른 손실 합 {o:.4f} (w 0: {b:.4f}, {100 * (o / b - 1):+.1f} %) · L_app {ap}', file=sys.stderr)
for n in ('ausvp_app', 'ausvp_app_w01'):
    if other(n)[0] <= b * 1.02: pick = n; break
print(pick)
PY
)
echo "=== 고른 것: $RUN $(date +%H:%M:%S)"; echo $RUN > $F/chosen.txt
echo "=== 2만 스텝 7명 학습 $(date +%H:%M:%S)"
POLITE=0 HOLD=0 OUT=$W/prism_full LP=prism_full STEPS=20000 SAVE=10000 GPU=0 RUNS=$RUN bash $PR/scripts/run_branch.sh
CK=$W/prism_full/$RUN/model_final.pt; [ -f $CK ] || { echo "학습 실패"; exit 1; }; C=$W/prism_full/eval_$RUN
steer() {  # $1 = 번호 (01 …): 조향 α → 생성 → 방향 보정 → 진단 렌더(조향 있음 · 없음) → 반응 · 질감 → Blender(조향 있음)
  local n=$1 S=sub-00$1
  $PY $PR/f1_steer_json.py --comp $C/infer/s$n/${S}_keep/components.npz --base $C/text/s$n/prompt.json --out $F/steer_s$n.json
  SUBN=s$n O8=$F TOK=$C/infer/s$n/${S}_keep/tokens P0=$C/text/s$n/prompt.json SEL=$W/prism_yaw/f1_s$n/yawsel.json STEERS="st_s$n=$F/steer_s$n.json" bash $PR/scripts/run_v8_steer.sh
  [ -d $F/diag/ns_s$n ] || (cd $MG && source $MG/mg_guard.sh >/dev/null && python $PR/v8_diag_render.py --glb $W/prism_yaw/f1_s$n/glb --out $F/diag/ns_s$n) 2>&1 | grep -E "DONE|Error"
  $PY -I $PR/v8_react_feat.py --diag $F/diag --sets ns_s$n,st_s$n --out $F/react | grep REACT
  $PY -I $PR/v10_render_tex.py --diag $F/diag --sets ns_s$n,st_s$n --out $F/react | grep TEX
  echo "DIAG_READY|s$n|$(date +%H:%M:%S)"
  local E=yaw_f1st_s$n; [ -f $V/out/$E/values.txt ] || bash $V/run_eval.sh $E $F/st_s$n/yawg glb 0 > $FE/logs/eval_$E.log 2>&1
  echo "CV|st_s$n | $(grep -E '^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(date +%H:%M:%S)"
}
echo "=== 피험자 1번 조향 없음 평가 $(date +%H:%M:%S)"
GPU=0 NS="01" CK=$CK CK_ROOT=$C TAG=f1 bash $PR/scripts/run_sub_eval.sh
echo "=== 피험자 1번 조향 $(date +%H:%M:%S)"; steer 01; echo "S01_DONE $(date +%H:%M:%S)"
echo "=== 나머지 6명 조향 없음 평가 $(date +%H:%M:%S)"
GPU=0 NS="02 03 05 06 07 08" CK=$CK CK_ROOT=$C TAG=f1 bash $PR/scripts/run_sub_eval.sh
for n in 02 03 05 06 07 08; do echo "=== s$n 조향 $(date +%H:%M:%S)"; steer $n; done
echo "=== F1_DONE $(date +%H:%M:%S)"
