#!/bin/bash
# 이미 만든 GLB 세트를 방향 보정 + 평가 (11-13, 5090) — GLB를 다른 서버에서 만들어 온 경우. GLB는 새로 만들지 않음
#  절차는 run_sub_eval.sh의 뒷부분과 같음: 12방위 렌더(512) → 방향 고르기(yawsel, 자기 토큰) → 회전 → 평가 기준 run_eval.sh
#  단계마다 개수 검사: 12방위 렌더 png = 104 × 12, yawsel.json 항목 = 104, 회전한 GLB = 104. 안 맞으면 그 세트는 멈춤 (값을 내지 않음)
#   — 11-13 Pro6000에서 렌더가 끊긴 채 다음 단계로 넘어가 방향 보정 없이 평가된 일이 있어서 넣음
#  평가의 외형 지표(tex.py, GPU 약 20 GB)가 메모리 부족으로 빠지면 GPU가 빌 때까지 기다렸다가 그 단계만 다시 계산
#  SETS="이름:GLB 폴더:토큰 폴더 …"  → $W/prism_yaw/<이름>, 값 fair_eval/fair_eval/out/yaw_<이름>/values.txt   세트는 차례로
# 사용: GPU=0 SETS="braup_s05:/…/glb_text:/…/tokens" nohup bash run_glb_eval.sh > $W/logs/glb_eval.log 2>&1 < /dev/null & disown
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; F=/home/oem/jhj/fair_eval; V=$F/fair_eval; Y=$W/prism_yaw
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home
free() { nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits; }
freewait() { until [ "$(free $CUDA_VISIBLE_DEVICES)" -ge $1 ]; do sleep 30; done; }
cd $W/run
for spec in ${SETS:?}; do
  IFS=: read n g t <<< "$spec"; O=$Y/$n; E=yaw_$n; mkdir -p $O
  ng=$(ls $g/*/*.glb 2>/dev/null | wc -l); nt=$(ls $t 2>/dev/null | wc -l)
  echo "=== $n 시작 $(date +%H:%M:%S) | GLB $ng · 토큰 $nt"
  [ $ng -eq 104 ] || { echo "!!! $n GLB가 104개가 아님 ($ng) — 건너뜀"; continue; }
  if [ ! -f $O/yawsel.json ]; then
    ls $g/*/*.glb > $O/list.txt
    $FE_BLENDER -b -t 4 -P $V/scripts/render.py -- --list $O/list.txt --out_root $O/render --res 512 --light 8000 --camera_dist 1.5 --fov 56 \
      --views $(seq -s, 0 30 330 | sed 's/\([0-9]*\)/\1:30/g') --metallic0 2>&1 | grep -E "FAIL" | head -5
    np=$(find $O/render -name "*.png" | wc -l); echo "CHK|$n|12방위 렌더 png $np / 1248 (물체 $(ls $O/render | wc -l))|$(date +%H:%M:%S)"
    [ $np -eq 1248 ] || { echo "!!! $n 12방위 렌더가 모자람 ($np) — 멈춤"; continue; }
    freewait 6000; trel yawsel --frames $O/render --pred $t --out $O --nview 12 | grep -E "YAWSEL_DONE|Error|없음"
  fi
  ns=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" $O/yawsel.json 2>/dev/null); echo "CHK|$n|yawsel 항목 ${ns:-0} / 104"
  [ "${ns:-0}" -eq 104 ] || { echo "!!! $n yawsel 항목이 104가 아님 — 멈춤"; continue; }
  [ $(ls $O/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/yaw_apply.py --glb $g --sel $O/yawsel.json --out $O/glb | grep YAWAPPLY
  na=$(ls $O/glb/*/*.glb 2>/dev/null | wc -l); [ $na -eq 104 ] || { echo "!!! $n 회전한 GLB $na / 104 — 멈춤"; continue; }
  [ -f $V/out/$E/values.txt ] || bash $V/run_eval.sh $E $O/glb glb $GPU > $F/logs/eval_$E.log 2>&1
  if [ ! -f $V/out/$E/values.txt ] && [ $(ls -d $V/out/$E/render_R2/* 2>/dev/null | wc -l) -ge 104 ]; then   # 외형 지표 단계가 빠진 경우
    echo "RETRY|$n|외형 지표 다시 (GPU 여유 대기) $(date +%H:%M:%S)"; freewait 21000
    (cd $V && $FE_PY $V/scripts/tex.py --pred $V/out/$E/render_R2 --gt $V/gt_render/R2 --keys $V/out/$E/keys.txt --out $V/out/$E/tex_R2.json > $V/out/$E/logs/tex_R2.log 2>&1) \
      && FE_SKIP_TEX=1 bash $V/run_eval.sh $E $O/glb glb $GPU >> $F/logs/eval_$E.log 2>&1
  fi
  echo "CV|$n | 평가 렌더 물체 $(ls -d $V/out/$E/render_R2/* 2>/dev/null | wc -l) | $(grep -E '^frame|^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(grep -o '물체별 최소 평균 [0-9.]*' $V/out/$E/logs/frame.log 2>/dev/null) | $(date +%H:%M:%S)"
done
echo "=== GLB_EVAL_DONE $(date +%H:%M:%S)"
