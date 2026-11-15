#!/bin/bash
# 최종 Ours Ablation 7행 × 피험자 7명 (11-15, 사용자: 다른 논문처럼 7명 평균, 행 하나씩 채움) — 5090만으로 (Pro6000은 미정)
#  학습: run_abl_final.sh와 같은 정의 · 설정 (7명 · 1,300개 · 검증 없음 · 2만, 갈래 손실 끄기)
#  평가: 한 행(모델) = 피험자 7명 run_sub_eval과 같은 절차 (텍스트판 GLB 104 → 12방위 렌더 → yawsel → 회전 → run_eval 8지표), TAG=abl_<이름>
#  GPU 하나 = 일꾼 하나. 일꾼은 PLAN에 적힌 순서대로: train:<이름>[+<이름>] (동시 학습) 또는 eval (공용 대기열에서 학습 끝난 행을 하나 집어 7명 평가)
#  안전장치 (오늘 끊김 대응): GLB 전 GPU 여유 21 GB 대기, yawsel 전 6 GB 대기, 렌더 수 검사(1248 · 624) 모자라면 다시, 외형 지표 실패면 여유 22 GB 기다렸다 다시 (최대 2번)
#   한 GPU에서 뒤에서 도는 렌더 · 평가(post)는 최대 MAXPOST개 (CPU 24코어, Blender -t 4)
#  PLAN 단계: train:a+b = 동시 학습하고 기다림 / bgtrain:a = 뒤에서 학습만 걸고 바로 다음 단계 / eval = 행 하나 평가
# 사용: GPU=0 PLAN="train:noS+noV train:noP+onlyS eval eval eval eval" WAIT_DIR=… WAIT_N=… nohup bash run_abl_rows.sh > $W/logs/abl_rows_g0.log 2>&1 < /dev/null & disown
W=/home/oem/jhj/MinD-3D++; SC=$W/mind_neural3d/prism_fmri/scripts; O=$W/prism_full_abl; Q=$O/queue; F=/home/oem/jhj/fair_eval; V=$F/fair_eval
ORDER=${ORDER:-"onlyV onlyP noS noV noSVP noP onlyS"}; NS="01 02 03 05 06 07 08"; MAXPOST=${MAXPOST:-3}
EU="--enc_path $W/weights/LEA_Visual_UKB/ukb_encoder.pt --in_norm 1 --feat_norm 1 --vit_lin 32 --mem branch --v_pool 2 --sv_enc 0"
flags() { case $1 in noS) echo "--lam_proto 0";; noV) echo "--lam_var 0";; noP) echo "--lam_subj 0 --film 0";;
                     onlyS) echo "--lam_var 0 --lam_subj 0 --film 0";; onlyV) echo "--lam_proto 0 --lam_subj 0 --film 0";; onlyP) echo "--lam_proto 0 --lam_var 0";;
                     noSVP) echo "--lam_proto 0 --lam_var 0 --lam_subj 0 --film 0";; *) echo BAD;; esac; }
GPU=${GPU:?}; source $SC/common.sh >/dev/null || exit 1; export CUDA_VISIBLE_DEVICES=$GPU
PY=$W/venv/bin/python; VT=$TGT/view_targets_shape.npz; TM=microsoft/TRELLIS-text-base; Y=$W/prism_yaw
export FE_PY=$F/venv/bin/python FE_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender TORCH_HOME=$F/torch_home
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
freewait() { until [ "$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits)" -ge $1 ]; do sleep 30; done; }
mkdir -p $O $Q; cd $W/run
[ -n "$WAIT_DIR" ] && { echo "=== 대기 $WAIT_DIR ${WAIT_N}개 $(date +%H:%M:%S)"; until [ $(ls $WAIT_DIR/*/*.glb 2>/dev/null | wc -l) -ge ${WAIT_N:-104} ]; do sleep 60; done; sleep 120; }

train() { local n=$1; [ -f $O/$n/model_final.pt ] && return
  echo "=== 학습 $n ($(flags $n)) $(date +%H:%M:%S)"
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008 --bs 14 --max_step 20000 \
    --per_subject 1 --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 --seed 1 --holdout 0 $EU $(flags $n) --resume --out_dir $O/$n > $W/logs/abl_final_$n.log 2>&1
  echo "=== 학습 끝 $n $(date +%H:%M:%S) $(ls $O/$n/model_final.pt 2>/dev/null)"; }

post() {   # $1 이름 $2 피험자: 방향 보정 → 평가 (검사 · 다시)
  local n=$1 s=$2 S=sub-00$2; local C=$O/eval_$n; local I=$C/infer/s$s g=$C/text/s$s/glb Yd=$Y/abl_${n}_s$s E=yaw_abl_${n}_s$s
  mkdir -p $Yd
  for t in 1 2 3; do
    [ -f $Yd/yawsel.json ] && break
    ls $g/*/*.glb > $Yd/list.txt
    $FE_BLENDER -b -t 4 -P $V/scripts/render.py -- --list $Yd/list.txt --out_root $Yd/render --res 512 --light 8000 --camera_dist 1.5 --fov 56 \
      --views $(seq -s, 0 30 330 | sed 's/\([0-9]*\)/\1:30/g') --metallic0 2>&1 | grep -E "FAIL" | head -3
    k=$(find $Yd/render -name '*.png' | wc -l); [ $k -ge 1248 ] || { echo "R!|$n s$s 방향 렌더 $k/1248 → 다시 ($t)"; continue; }
    freewait 6000; trel yawsel --frames $Yd/render --pred $I/${S}_keep/tokens --out $Yd --nview 12 | grep -E "YAWSEL_DONE|Error|없음"
  done
  [ $(ls $Yd/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/yaw_apply.py --glb $g --sel $Yd/yawsel.json --out $Yd/glb | grep YAWAPPLY
  for t in 1 2 3; do
    grep -q "R2|LPIPS" $V/out/$E/values.txt 2>/dev/null && break
    [ $t -gt 1 ] && freewait 22000
    bash $V/run_eval.sh $E $Yd/glb glb $GPU > $F/logs/eval_$E.log 2>&1
    k=$(find $V/out/$E/render_R2 -name '*.png' 2>/dev/null | wc -l); grep -q "R2|LPIPS" $V/out/$E/values.txt 2>/dev/null || echo "E!|$n s$s 평가 렌더 $k/624 · 외형 지표 없음 → 다시 ($t)"
  done
  echo "CV|$n s$s | $(grep -E '^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^(CD|EMD|FPD)\s' $V/out/$E/values.txt 2>/dev/null | awk '{printf "%s ", $2}') | $(date +%H:%M:%S)"
}

evalrow() {   # 대기열에서 학습 끝난 행 하나를 집어 7명 평가
  local n=""
  while [ -z "$n" ]; do
    for c in $ORDER; do [ -f $O/$c/model_final.pt ] && mkdir $Q/$c 2>/dev/null && { n=$c; break; }; done
    [ -n "$n" ] && break
    left=0; for c in $ORDER; do [ -d $Q/$c ] || left=1; done; [ $left = 0 ] && { echo "=== 남은 행 없음"; return; }
    sleep 60
  done
  echo "=== 행 평가 $n $(date +%H:%M:%S)"; local C=$O/eval_$n
  for s in $NS; do
    local S=sub-00$s I=$C/infer/s$s g=$C/text/s$s/glb; mkdir -p $C/text/s$s
    [ -f $I/${S}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $O/$n/model_final.pt --out $I --subs $S --perceptive keep | grep -E "범주|DONE|Error"
    $PY $PR/text_prompts.py --comp $I/${S}_keep/components.npz --out $C/text/s$s/prompt.json | tail -1
    while [ $(jobs -rp | wc -l) -ge $MAXPOST ]; do sleep 30; done
    freewait 21000
    [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/${S}_keep/tokens --slat_text $C/text/s$s/prompt.json --text_model $TM --out $g | grep -E "GEN_DONE|Error|없음"
    echo "T|$n s$s|GLB $(ls $g/*/*.glb 2>/dev/null | wc -l)/104|$(date +%H:%M:%S)"
    post $n $s &
  done
}

for step in ${PLAN:?}; do
  case $step in
    train:*) names=${step#train:}; P=""; for n in ${names//+/ }; do train $n & P="$P $!"; sleep 60; done; wait $P;;
    bgtrain:*) n=${step#bgtrain:}; train $n > /dev/null & echo "=== 뒤에서 학습 $n (같은 GPU에서 평가와 함께, 학습 6 GB + GLB 21 GB)";;
    eval) evalrow;;
  esac
done
wait; echo "=== ABL_ROWS_DONE GPU$GPU $(date +%H:%M:%S)"
