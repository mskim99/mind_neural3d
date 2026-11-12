#!/bin/bash
# 판정 1번 형상 출처 비교 (11-12, 5090): Subject 1 test 104, 외형은 모두 텍스트판 "a {g0 예측 범주}"(prompt_vt_s1.json), 형상만 바꿈
#  1a  범주 조건 prior(_cL, CFG 3), 범주 = g0 예측          1b  같은 prior, 범주 = 정답 (상한)
#  1c_t{03,05,07}  g0 토큰 → TRELLIS 1단계 z(gen과 같은 noise · seed 42) → t_start 0.3/0.5/0.7까지 노이즈 → 같은 prior(예측 범주, CFG 3)로 되돌림
#  행마다: 형상 잠재 → glb (빈 형상은 그 물체만 seed 바꿔 다시, 정답 안 봄) → [GPU의 행 GLB가 다 끝난 뒤] 방향 보정(12방위 렌더 → yawsel(g0 토큰) → 회전) → 평가 기준 run_eval.sh
#  GLB 생성과 평가를 같은 GPU에서 겹치지 않게 함 (tex.py 약 10 GB + TRELLIS 약 21 GB > 32 GB)
#  출력: $J/<행>/{z,glb}, $W/prism_yaw/judge_<행>/glb, 값 $FE/out/yaw_judge_<행>/values.txt (물체마다 방향을 맞춘 CD = logs/frame.log '물체별 최소 평균', 진단용)
# 사용: GPU=0 ROWS="1a 1b c03" nohup bash run_judge_shape.sh > $W/logs/judge_shape_g0.log 2>&1 < /dev/null & disown
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
ROWS=${ROWS:?}; PY=$W/venv/bin/python; J=$W/prism_judge; CK=$W/prism_b/ss_prior_cL/prior_final.pt; SS=$W/prism_b/ss
I=$W/prism_vshape/infer/vshape_g0/sub-0001_keep; TK=$I/tokens; PROMPT=$W/prism_vshape/text/prompt_vt_s1.json
F=/home/oem/jhj/fair_eval; V=$F/fair_eval; Y=$W/prism_yaw; CFG=${CFG:-3.0}; PAR=${PAR:-2}
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
st() { echo "T|$1|$2|$(( $(date +%s) - $3 ))|$(date +%H:%M:%S)"; }
mkdir -p $J; cd $W/run
[ -f $J/pred_cls.json ] || $PY -c "
import numpy as np, json; z = np.load('$I/components.npz'); C = [str(c) for c in z['classes']]; N = [str(k).split('/')[-1] for k in z['name']]
json.dump({n: C[int(c)] for n, c in zip(N, z['cls_pred'])}, open('$J/pred_cls.json', 'w'), indent=1)
json.dump({n: C[int(c)] for n, c in zip(N, z['cls'])}, open('$J/gt_cls.json', 'w'), indent=1)
print('물체', len(N), '| g0 범주 정확도', round(float((z['cls'] == z['cls_pred']).mean()), 3))"
gen_row() {   # 행 이름
  local r=$1 D=$J/$1 t=$(date +%s) cls=$J/pred_cls.json X=""
  case $r in 1b) cls=$J/gt_cls.json;; c0[0-9]) X="--init_z $J/g0_z/z --t_start 0.${r#c0}";; esac
  if [ -n "$X" ] && [ $(ls $J/g0_z/z/*.npy 2>/dev/null | wc -l) -lt 104 ]; then
    if mkdir $J/g0_z.lock 2>/dev/null; then   # 두 GPU가 같이 만들지 않게 — 먼저 잡은 쪽만 만들고 다른 쪽은 기다림
      trel ss --pred $TK --out $J/g0_z --bs 1 --seed 42 --ss_noise cpu | grep -E "SS_DONE|Error"
    fi
    until [ $(ls $J/g0_z/z/*.npy 2>/dev/null | wc -l) -ge 104 ]; do sleep 10; done
  fi
  mkdir -p $D
  [ $(ls $D/z/*.npy 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/prior_ss.py sample --ss $SS --ckpt $CK --out $D/z --cls_json $cls --temp 1.0 --cfg $CFG $X | grep -E "생성|Error"
  trel gen --ss_z $D/z --slat_text $PROMPT --out $D/glb | grep -E "GEN_DONE|Error|빈 형상"
  for k in 1 2 3; do
    $PY -c "
import json, os; J = json.load(open('$cls')); M = {n: c for n, c in J.items() if not os.path.exists('$D/glb/'+n+'/'+n+'.glb')}
json.dump(M, open('$D/retry.json', 'w')); print('빈 형상 다시 뽑기', len(M))"
    [ "$(cat $D/retry.json)" = "{}" ] && break
    $PY $PR/prior_ss.py sample --ss $SS --ckpt $CK --out $D/z --cls_json $D/retry.json --temp 1.0 --cfg $CFG --seed $k $X | grep -E "생성|Error"
    trel gen --ss_z $D/z --slat_text $PROMPT --out $D/glb | grep -E "GEN_DONE|Error|빈 형상"
  done
  st $r GLB_$(ls $D/glb/*/*.glb | wc -l)개 $t
}
eval_row() {
  local r=$1 g=$J/$1/glb O=$Y/judge_$1 E=yaw_judge_$1 t=$(date +%s)
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || { echo "건너뜀 $r (GLB $(ls $g/*/*.glb 2>/dev/null | wc -l))"; return; }
  mkdir -p $O
  if [ ! -f $O/yawsel.json ]; then
    ls $g/*/*.glb > $O/list.txt
    $FE_BLENDER -b -t 4 -P $V/scripts/render.py -- --list $O/list.txt --out_root $O/render --res 512 --light 8000 --camera_dist 1.5 --fov 56 \
      --views $(seq -s, 0 30 330 | sed 's/\([0-9]*\)/\1:30/g') --metallic0 2>&1 | grep -E "FAIL" | head -5
    trel yawsel --frames $O/render --pred $TK --out $O --nview 12 | grep -E "YAWSEL_DONE|Error|없음"
    st $r 방향고르기 $t; t=$(date +%s)
  fi
  [ $(ls $O/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/yaw_apply.py --glb $g --sel $O/yawsel.json --out $O/glb | grep YAWAPPLY
  [ -f $V/out/$E/values.txt ] || bash $V/run_eval.sh $E $O/glb glb $GPU > $F/logs/eval_$E.log 2>&1
  st $r 평가 $t
  echo "JV|$r | $(grep -E '^frame|^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(grep -o '물체별 최소 평균 [0-9.]*' $V/out/$E/logs/frame.log 2>/dev/null)"
}
export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home FE_WORKERS=${FE_WORKERS:-8} FE_THREADS=${FE_THREADS:-6}
for r in $ROWS; do gen_row $r; done
echo "=== GLB 끝 $(date +%H:%M:%S)"
for r in $ROWS; do while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 20; done; eval_row $r & sleep 5; done
wait; echo "=== JUDGE_SHAPE_DONE $(date +%H:%M:%S)"
