#!/bin/bash
# V-8 (11-28, 5090): 외형 단계가 어떤 조건에 반응하나 — 형상은 고정, 외형 조건만 바꿔 다시 생성 → 평가 기준 (학습 없음)
#  형상: 기존 Ours(7명 학습, prism_full/ausvp) 피험자 1번 test 104개 예측 토큰(prism_full/eval_ausvp/infer/s01/sub-0001_keep/tokens) → TRELLIS 1단계, seed 42 (기준 ㉠과 같은 명령)
#  외형: 세트마다  이름|text|<문장 json>  = 텍스트판 SLat flow (기준 ㉠과 같은 경로, 문장만 바꿈)
#                 이름|img|<토큰 폴더>   = 이미지판 SLat flow, 조건 = 폴더의 DINOv2 토큰 (trellis_tools --slat_pred)
#  방향 보정: 기준 ㉠(yaw_fullausvp_s01)의 yawsel.json을 그대로 씀 (형상이 같으므로 세트마다 다시 고르지 않음)
#  평가: run_eval.sh (8지표). 생성은 GPU 한 번에 하나(gen.lock), 평가는 CPU 한 번에 하나(eval.lock) — 여러 번 불러도 줄을 섬
# 사용 (절대 경로로): SETS="gt_detail|text|$W/prism_v8/prompts/gt_detail.json …" nohup bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/run_v8.sh > $W/logs/v8_<묶음>.log 2>&1 < /dev/null &
GPU=0 source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
GPU=0   # 11-28: source 앞 임시 지정은 끝나면 사라짐 → run_eval에 빈 GPU 번호가 넘어가 렌더가 CPU로 돌고 텍스처 지표가 실패했음
PY=$W/venv/bin/python; TM=microsoft/TRELLIS-text-base; F=/home/oem/jhj/fair_eval; V=$F/fair_eval; O=$W/prism_v8; mkdir -p $O
export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home
TOK=$W/prism_full/eval_ausvp/infer/s01/sub-0001_keep/tokens; SEL=$W/prism_yaw/fullausvp_s01/yawsel.json
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
freewait() { until [ "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 30; done; }
ev() {   # $1 = 이름
  local n=$1 E=yaw_v8_$1
  [ $(ls $O/$n/yaw/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/yaw_apply.py --glb $O/$n/glb --sel $SEL --out $O/$n/yaw | grep YAWAPPLY
  [ -f $V/out/$E/values.txt ] || bash $V/run_eval.sh $E $O/$n/yaw glb $GPU > $F/logs/eval_$E.log 2>&1
  echo "CV|$n | $(grep -E '^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | tr -s ' \t\n' ' ') | $(date +%H:%M:%S)"
}
cd $W/run
for s in ${SETS:?}; do
  IFS='|' read n k arg <<< "$s"; mkdir -p $O/$n
  (
    flock 8
    wait_gpu_free v8_$n; freewait 19000
    echo "=== $n 생성 시작 ($k) $(date +%H:%M:%S)"
    if [ $(ls $O/$n/glb/*/*.glb 2>/dev/null | wc -l) -lt 104 ]; then
      if [ $k = text ]; then trel gen --pred $TOK --slat_text $arg --text_model $TM --out $O/$n/glb | grep -E "GEN_DONE|Error|없음|같음"
      else trel gen --pred $TOK --slat_pred $arg --out $O/$n/glb | grep -E "GEN_DONE|Error|없음"; fi
    fi
    echo "T|$n|GLB $(ls $O/$n/glb/*/*.glb 2>/dev/null | wc -l)개|$(date +%H:%M:%S)"
  ) 8>$O/gen.lock
  ( flock 9; ev $n ) 9>$O/eval.lock &
done
wait
echo "=== V8_DONE $(date +%H:%M:%S)"
