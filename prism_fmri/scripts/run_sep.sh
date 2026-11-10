#!/bin/bash
# S · V · P 분리 (11-10, 사용자 승인, Pro6000 GPU 1) — 설계 `.보고 자료/261112_9차보고/PRISM_분리/분리·텍스트_설계.md`
#  기준 = vshape_g0 설정(ablation full과 같음: fig 7명 bs 14 4만 스텝, V = 범주 안 개체 형상, GRL 없음) + 분리 손실 하나
#    tc05 : --lam_tc 0.5   (A. 결합-섞기 판별기, sep_losses.py)
#    tc01 : --lam_tc 0.1
#    hsic1: --lam_hsic 1.0 (C. 갈래 쌍 RBF CKA 합, 대조)
#  0 기준 vshape_g0: 7명 test 추론(성분만) → 판정(sep_eval.py)        ← 학습 없음, 먼저
#  1 RUNS를 이 GPU에서 동시에 학습 → 끝난 것부터 7명 test 추론 → 판정 → sub-0001 GLB(TRELLIS 이미지판) → 공정평가 8지표
#  → $W/prism_sep/sep.tar (SEP_DONE). 판정 줄 SEP|…, 지표 $FE/out/sep_<이름>/values.txt
# 사용: nohup bash run_sep.sh > $W/logs/prism_sep.log 2>&1 < /dev/null & disown      (RUNS="tc05 hsic1"처럼 고를 수 있음)
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
RUNS=${RUNS:-"tc05 tc01 hsic1"}
FE=${FE:-$W/fair_eval}; PY=$W/venv/bin/python; ENC=${ENC:-$W/weights/MinD-3D-pp/fMRI-Shape}; VT=$TGT/view_targets_shape.npz
O=$W/prism_sep; S1=sub-0001; OTH=sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
export FE_PY=$PY FE_WORKERS=${FE_WORKERS:-6} TORCH_HOME=$W/cache/torch FE_BLENDER=${FE_BLENDER:-$W/tools/blender-4.2.3-linux-x64/blender}
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
step() { echo "=== $1 $(date +%H:%M:%S)"; }
flags() { case $1 in tc05) echo "--lam_tc 0.5";; tc01) echo "--lam_tc 0.1";; hsic1) echo "--lam_hsic 1.0";; *) echo BAD;; esac; }
for n in $RUNS; do [ "$(flags $n)" = BAD ] && { echo "모르는 이름 $n"; exit 1; }; done
[ -f $VT ] && [ -f $FE/run_temp.sh ] || { echo "없음: $VT 또는 $FE/run_temp.sh"; exit 1; }
cd $W/run; mkdir -p $O/judge

judge() {   # 이름 · 체크포인트 → 7명 test 추론(sub-0001만 토큰 저장) → 판정
  local n=$1 ck=$2 I=$O/infer/$1
  [ -f $I/${S1}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $ck --out $I --subs $S1 --perceptive keep | grep -E "범주|Error"
  [ -f $I/sub-0008_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $ck --out $I --subs $OTH --perceptive keep --save_tokens 0 | grep -E "범주|Error"
  [ -f $O/judge/$n.json ] || $PY $PR/sep_eval.py --comp $(for s in ${SUBS//,/ }; do echo $I/${s}_keep/components.npz; done) --vt $VT --out $O/judge/$n.json --name $n | grep -E "^SEP|Error"
}

one() {   # 학습 → 판정 → GLB → 8지표
  local n=$1 M=$O/$1 I=$O/infer/$1
  if [ ! -f $M/model_final.pt ]; then
    echo "=== 학습 시작 $n ($(flags $n)) $(date +%H:%M:%S)"
    $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step 40000 \
      --per_subject 1 --enc_dir $ENC --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 $(flags $n) --resume \
      --out_dir $M > $W/logs/prism_sep_$n.log 2>&1
    echo "=== 학습 끝 $n $(date +%H:%M:%S) | $(grep -E '^\[val\]' $W/logs/prism_sep_$n.log | tail -1 | cut -c1-260)"
  fi
  [ -f $M/model_final.pt ] || { echo "체크포인트 없음 $n (로그 logs/prism_sep_$n.log)"; return 1; }
  judge $n $M/model_final.pt
  [ $(ls $I/${S1}_keep/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/${S1}_keep/tokens --out $I/${S1}_keep/glb | grep -E "GEN_DONE|Error"
  echo "GLB $n $(ls $I/${S1}_keep/glb/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ -f $FE/out/sep_$n/values.txt ] || bash $FE/run_temp.sh sep_$n $I/${S1}_keep/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/fe_sep_$n.log 2>&1
  echo "EVAL_DONE sep_$n $(date +%H:%M:%S) | $(grep -E '^R2\|10way|^FPD\s|^P2\|CD\s' $FE/out/sep_$n/values.txt 2>/dev/null | tr '\n' ' ')"
}

step "1 학습 시작 (동시) + 기준 vshape_g0 판정"
J=""; for n in $RUNS; do one $n & J="$J $!"; sleep 90; done
step "0 기준 vshape_g0 판정 (학습 없음)"
judge g0 $W/prism_vshape/vshape_g0/model_final.pt
wait $J

step "2 묶기"
P=$O/pack; mkdir -p $P; cp $O/judge/*.json $P/
for n in $RUNS; do
  mkdir -p $P/$n; cp $O/$n/train_log.csv $W/logs/prism_sep_$n.log $P/$n/ 2>/dev/null; cp $FE/out/sep_$n/values.txt $P/$n/ 2>/dev/null
  cp $O/infer/$n/${S1}_keep/components.npz $P/$n/ 2>/dev/null
done
cd $O && tar cf sep.tar pack && echo "PACK $(tar tf sep.tar | wc -l)개"
echo "=== 결과"; for n in g0 $RUNS; do echo "$n | $(grep -E '^R2\|(2way|10way)|^FPD\s|^P2\|(CD|EMD)\s' $FE/out/sep_$n/values.txt 2>/dev/null | tr '\n' ' ')"; done
step "SEP_DONE"
