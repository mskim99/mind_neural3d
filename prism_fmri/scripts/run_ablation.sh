#!/bin/bash
# Ablation (11-10, 사용자: 현재 버전 vshape_g0에서 갈래를 하나씩 · 다 빼기, Subject 1, FPD · CD · EMD만)
#  "뺀다" = 그 갈래를 가르치는 손실을 끔 (구조는 그대로):
#    noS   : --lam_proto 0                         (Semantic CLIP 프로토타입 손실)
#    noV   : --lam_var 0                           (Variational 개체 형상 손실)
#    noP   : --lam_subj 0 --film 0                 (Perceptive 피험자 분류 손실 + FiLM 보정)
#    noSVP : 위 셋 모두
#  나머지 설정은 vshape_g0과 같음 (run_vshape.sh: fig 7명 bs 14 4만 스텝, V = 범주 안 개체 형상, GRL 없음)
#  ABL에 적은 것을 이 GPU에서 동시에 학습 → 끝난 것부터 sub-0001 test 추론 → GLB(TRELLIS 이미지판) → fe_shape
#  → $FE/out/abl_<이름>/values.txt, 묶음 $W/prism_abl/abl_<ABL 이름들>.tar (ABL_DONE)
# 사용 (Pro6000): ABL="noP noSVP" nohup bash run_ablation.sh > $W/logs/prism_abl.log 2>&1 < /dev/null & disown
#   5090 등 다른 서버: common.sh 경로만 그 서버에 맞추고 GPU=0 ABL=noS … / GPU=1 ABL=noV … 처럼 GPU마다 따로 띄움
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
source $(dirname $0)/fe_shape.sh
ABL=${ABL:?ABL="full noS noV noP noSVP" 중 고르기}
SEED=${SEED:--1}; SUF=$([ "$SEED" -ge 0 ] && echo "_s$SEED")   # 11-10: SEED ≥ 0이면 --seed + 이름 뒤 _s<SEED>
EVAL=${EVAL:-shape}   # shape = FPD · CD · EMD만 / full = 공정평가 8지표 (run_temp.sh, 렌더 포함)
FE=${FE:-$W/fair_eval}; PY=$W/venv/bin/python; ENC=${ENC:-$W/weights/MinD-3D-pp/fMRI-Shape}; VT=${VT:-$TGT/view_targets_shape.npz}
O=$W/prism_abl; S1=sub-0001
export FE_PY=$PY FE_WORKERS=${FE_WORKERS:-6} TORCH_HOME=$W/cache/torch
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
[ -f $VT ] && [ -f $FE/scripts/dist.py ] || { echo "없음: $VT 또는 공정평가 번들 $FE"; exit 1; }
flags() { case $1 in full) echo "";; noS) echo "--lam_proto 0";; noV) echo "--lam_var 0";; noP) echo "--lam_subj 0 --film 0";;
                     noSVP) echo "--lam_proto 0 --lam_var 0 --lam_subj 0 --film 0";; *) echo BAD;; esac; }
for n in $ABL; do [ "$(flags $n)" = BAD ] && { echo "모르는 이름 $n"; exit 1; }; done
cd $W/run; mkdir -p $O
one() {   # 이름: 학습 → 추론 → GLB → 형태 지표
  local n=$1 M=$O/$1$SUF I=$O/infer/$1$SUF
  if [ ! -f $M/model_final.pt ]; then
    echo "=== 학습 시작 $n ($(flags $n)) $(date +%H:%M:%S)"
    $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step 40000 \
      --per_subject 1 --enc_dir $ENC --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 $(flags $n) --seed $SEED --resume \
      --out_dir $M > $W/logs/prism_abl_$n$SUF.log 2>&1
    echo "=== 학습 끝 $n $(date +%H:%M:%S) | $(grep -E '^\[val\]' $W/logs/prism_abl_$n$SUF.log | tail -1 | cut -c1-200)"
  fi
  [ -f $M/model_final.pt ] || { echo "체크포인트 없음 $n (로그 logs/prism_abl_$n.log)"; return 1; }
  [ -f $I/${S1}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $M/model_final.pt --out $I --subs $S1 --perceptive keep | grep -E "범주|DONE|Error"
  [ $(ls $I/${S1}_keep/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/${S1}_keep/tokens --out $I/${S1}_keep/glb | grep -E "GEN_DONE|Error"
  echo "GLB $n $(ls $I/${S1}_keep/glb/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  if [ "$EVAL" = full ]; then
    export FE_BLENDER=${FE_BLENDER:-$W/tools/blender-4.2.3-linux-x64/blender}
    bash $FE/run_temp.sh abl8_$n$SUF $I/${S1}_keep/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/abl8_$n$SUF.log 2>&1; echo "EVAL_DONE abl8_$n$SUF $(date +%H:%M:%S)"
  else
    fe_shape abl_$n$SUF $I/${S1}_keep/glb glb $DR/annotations/core_test_list.txt
  fi
}
J=""; for n in $ABL; do one $n & J="$J $!"; sleep 90; done
wait $J
echo "=== 결과"; for n in $ABL; do echo "$n$SUF $(grep -E '^P2\|(CD|EMD)\s|^FPD\s|^R2\|(2way|10way)\s' $FE/out/abl_$n$SUF/values.txt $FE/out/abl8_$n$SUF/values.txt 2>/dev/null | tr '\n' ' ')"; done
T=$O/abl_$(echo $ABL | tr ' ' '_')$SUF.tar
tar cf $T -C $FE $(cd $FE && for n in $ABL; do ls -d out/abl_$n$SUF/values.txt out/abl8_$n$SUF/values.txt 2>/dev/null; done) 2>/dev/null
tar rf $T -C $W $(for n in $ABL; do echo logs/prism_abl_$n$SUF.log; done) 2>/dev/null
echo "PACK $T $(tar tf $T 2>/dev/null | wc -l)개"
echo "=== ABL_DONE $(date +%H:%M:%S)"
