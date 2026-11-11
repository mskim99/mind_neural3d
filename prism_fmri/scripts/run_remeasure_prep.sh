#!/bin/bash
# 지표 재측정 준비 (11-11, 사용자): 평가 방식(4090, EMD 처리)과 무관한 학습 · 추론 · GLB 생성만 먼저. 평가는 안 함
#  계획 `.보고 자료/261112_9차보고/재측정/재측정_분배.md`. 생성 설정은 Ours 기준과 같음: 형상 = 예측 토큰, 외형 = TRELLIS 텍스트판 "a {예측 범주}"
#  TRAIN : 새로 학습할 ablation (vshape_g0 설정에서 갈래 손실만 끔, run_ablation.sh와 같은 명령 · 같은 출력 폴더 $W/prism_abl/<이름>)
#          하나씩 넣은 것  onlyS = S 손실만 / onlyV = V 손실만 / onlyP = P 손실 + FiLM만
#          (있는 것) full · noS · noV · noP · noSVP
#  ABLG  : sub-0001 test 104 텍스트판 GLB를 만들 ablation 이름 (적은 순서대로, 체크포인트가 학습 중이면 끝날 때까지 기다림)
#          → $W/prism_abl/infer/<이름>/sub-0001_keep/glb_text
#  SUBG  : Ours(vshape_g0 시드 1) 텍스트판 GLB를 만들 피험자 → $W/prism_vshape/text/glb/vt_s<번호> (vt_s1 · vt_s8은 이미 있음)
#  학습 · 피험자 GLB · ablation GLB 세 줄이 같은 GPU에서 동시에 돎
# 사용 (Pro6000 GPU 1): TRAIN="onlyP" ABLG="noSVP noP onlyP" SUBG="sub-0002 sub-0003 sub-0005 sub-0006 sub-0007" nohup bash run_remeasure_prep.sh > $W/logs/prism_prep.log 2>&1 < /dev/null & disown
#      (5090 GPU마다):   GPU=0 TRAIN="onlyS" ABLG="onlyS noS" SUBG="" …  /  GPU=1 TRAIN="onlyV" ABLG="onlyV noV" SUBG="" …
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
TRAIN=${TRAIN:-}; ABLG=${ABLG:-}; SUBG=${SUBG:-}
PY=$W/venv/bin/python; ENC=${ENC:-$W/weights/MinD-3D-pp/fMRI-Shape}; VT=${VT:-$TGT/view_targets_shape.npz}; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}
O=$W/prism_abl; S1=sub-0001; CK0=$W/prism_vshape/vshape_g0/model_final.pt; I0=$W/prism_vshape/infer/vshape_g0; OT=$W/prism_vshape/text
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cnt() { ls $1/*/*.glb 2>/dev/null | wc -l; }
flags() { case $1 in full) echo "";; noS) echo "--lam_proto 0";; noV) echo "--lam_var 0";; noP) echo "--lam_subj 0 --film 0";;
                     noSVP) echo "--lam_proto 0 --lam_var 0 --lam_subj 0 --film 0";;
                     onlyS) echo "--lam_var 0 --lam_subj 0 --film 0";; onlyV) echo "--lam_proto 0 --lam_subj 0 --film 0";; onlyP) echo "--lam_proto 0 --lam_var 0";; *) echo BAD;; esac; }
for n in $TRAIN $ABLG; do [ "$(flags $n)" = BAD ] && { echo "모르는 이름 $n"; exit 1; }; done
[ -f $VT ] || { echo "없음: $VT"; exit 1; }
cd $W/run; mkdir -p $O $OT/glb

train() {   # 이름
  local n=$1 M=$O/$1
  [ -f $M/model_final.pt ] && { echo "학습 건너뜀 $n (이미 있음)"; return; }
  echo "=== 학습 시작 $n ($(flags $n)) $(date +%H:%M:%S)"
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step 40000 \
    --per_subject 1 --enc_dir $ENC --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 $(flags $n) --seed -1 --resume \
    --out_dir $M > $W/logs/prism_abl_$n.log 2>&1
  echo "=== 학습 끝 $n $(date +%H:%M:%S) | $(grep -E '^\[val\]' $W/logs/prism_abl_$n.log | tail -1 | cut -c1-200)"
}
textglb() {   # 표시 이름 · 추론 폴더(<…>_keep) · 문장 json · GLB 폴더
  [ -f $2/components.npz ] && [ $(ls $2/tokens/*.npy 2>/dev/null | wc -l) -ge 104 ] || { echo "예측 없음 $1 ($2)"; return 1; }
  $PY $PR/text_prompts.py --comp $2/components.npz --out $3 > /dev/null
  [ $(cnt $4) -ge 104 ] || trel gen --pred $2/tokens --slat_text $3 --text_model $TM --out $4 | grep -E "GEN_DONE|Error|없음"
  echo "GLB $1 $(cnt $4)개 $(date +%H:%M:%S)"
}
lane_sub() {
  for s in $SUBG; do
    local I=$I0/${s}_keep n=vt_s$((10#${s#sub-}))
    [ -f $I/components.npz ] && [ $(ls $I/tokens/*.npy 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK0 --out $I0 --subs $s --perceptive keep | grep -E "범주|Error"
    textglb "Ours $s" $I $OT/prompt_$n.json $OT/glb/$n
  done
}
lane_abl() {
  for n in $ABLG; do
    local M=$O/$n I=$O/infer/$n
    if [ ! -f $M/model_final.pt ]; then
      case " $TRAIN " in *" $n "*) until [ -f $M/model_final.pt ]; do sleep 120; done;; *) echo "체크포인트 없음 $n ($M) → 건너뜀"; continue;; esac
    fi
    [ -f $I/${S1}_keep/components.npz ] && [ $(ls $I/${S1}_keep/tokens/*.npy 2>/dev/null | wc -l) -ge 104 ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $M/model_final.pt --out $I --subs $S1 --perceptive keep | grep -E "범주|Error"
    textglb "ablation $n" $I/${S1}_keep $I/prompt_text.json $I/${S1}_keep/glb_text
  done
}
J=""; for n in $TRAIN; do train $n & J="$J $!"; sleep 90; done
lane_sub & J="$J $!"
lane_abl & J="$J $!"
wait $J
echo "=== 결과 (GLB 개수, 104여야 함)"
for s in sub-0001 $SUBG sub-0008; do n=vt_s$((10#${s#sub-})); echo "Ours $s $(cnt $OT/glb/$n)"; done
for n in $ABLG; do echo "ablation $n $(cnt $O/infer/$n/${S1}_keep/glb_text)"; done
echo "=== PREP_DONE $(date +%H:%M:%S)"
