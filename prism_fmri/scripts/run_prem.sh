#!/bin/bash
# Perceptive = 버리는 나머지 (11-11, 사용자 제안 · 설정 승인, 후순위 실행) — 설계 · 판정 기준 `.보고 자료/261112_9차보고/PRISM_분리/P나머지_설계.md`
#  기준 = vshape_g0 설정(fig 7명 bs 14 4만 스텝, V = 범주 안 개체 형상) + 공통 변경 세 가지:
#    ① 디코더에 P 안 넣음 (--film 0, 피험자 분류 손실은 P에 그대로)   ② 짝 배치 (물체 7 × 피험자 2)   ③ P = 나머지 복원 (--lam_rem 0.1)
#  RUNS (일치 손실 = 같은 물체 · 다른 피험자의 벡터를 가깝게, 온도 0.07):
#    s05 · s10   : S에만, 세기 0.5 · 1.0          sv05 · sv10 : S · V 둘 다, 세기 0.5 · 1.0
#    ctrl        : 일치 손실 없음 (짝 배치 + ① + ③만 — 일치 손실의 효과를 떼어 보는 대조군)
#    ctrl0       : ctrl에서 ③도 뺌 (짝 배치 + ①만) — ③은 S · V 갈래 모듈로는 기울기가 안 가지만 공유 Temporal Stem · SV Encoder로는 감 → ctrl 대 ctrl0로 ③의 영향을 잼
#  일치 손실 세기는 처음 2,000스텝 동안 0 → λ (--share_warm). [val] 줄의 '검색 S · V' = test에서 같은 물체를 다른 피험자 표본 중 1등으로 찾는 비율
#  RUNS를 이 GPU에서 동시에 학습 → 끝난 것부터: 7명 test 추론 → 분리 판정(sep_eval.py, 판정 3) → sub-0001 텍스트판 GLB(판정 1 입력)
#    → APAC=1이면 Subject 1 갈래에 sub-0009(AP) · sub-0011(APAC) fMRI를 넣어 GLB(판정 2 입력, 이미지판 = run_apac.sh와 같음)
#  평가(8지표 · 형태 지표)는 여기서 안 함 — 새 평가 번들로 따로. 판정 4(피험자별 test 범주 정확도)는 [val] 줄
#  → $W/prism_prem/<이름>, 판정 json $W/prism_prem/judge/<이름>.json (PREM_DONE)
# 사용: GPU=1 RUNS="s05 sv05" nohup bash run_prem.sh > $W/logs/prism_prem.log 2>&1 < /dev/null & disown
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
RUNS=${RUNS:-"s05 s10 sv05 sv10 ctrl ctrl0"}; APAC=${APAC:-1}
PY=$W/venv/bin/python; ENC=${ENC:-$W/weights/MinD-3D-pp/fMRI-Shape}; VT=$TGT/view_targets_shape.npz; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}
O=$W/prism_prem; S1=sub-0001; OTH=sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
declare -A FS=([AP]=sub-0009 [APAC]=sub-0011) LS=([AP]=$DR/annotations/core_test_list.txt [APAC]=$DR/annotations/apac_sub0011_list.txt) NN=([AP]=104 [APAC]=220)
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cnt() { ls $1/*/*.glb 2>/dev/null | wc -l; }
flags() { case $1 in s05) echo "--lam_rem 0.1 --lam_share 0.5 --share_on s";; s10) echo "--lam_rem 0.1 --lam_share 1.0 --share_on s";;
                     sv05) echo "--lam_rem 0.1 --lam_share 0.5 --share_on sv";; sv10) echo "--lam_rem 0.1 --lam_share 1.0 --share_on sv";;
                     ctrl) echo "--lam_rem 0.1 --paired 1";; ctrl0) echo "--lam_rem 0 --paired 1";; *) echo BAD;; esac; }
for n in $RUNS; do [ "$(flags $n)" = BAD ] && { echo "모르는 이름 $n"; exit 1; }; done
[ -f $VT ] || { echo "없음: $VT"; exit 1; }
cd $W/run; mkdir -p $O/judge

one() {
  local n=$1 M=$O/$1 I=$O/infer/$1
  if [ ! -f $M/model_final.pt ]; then
    echo "=== 학습 시작 $n ($(flags $n)) $(date +%H:%M:%S)"
    $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --bs 14 --max_step 40000 \
      --per_subject 1 --enc_dir $ENC --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv 0 \
      --film 0 --share_temp 0.07 --share_warm 2000 --pair_k 2 $(flags $n) --resume --out_dir $M > $W/logs/prism_prem_$n.log 2>&1
    echo "=== 학습 끝 $n $(date +%H:%M:%S) | $(grep -E '^\[val\]' $W/logs/prism_prem_$n.log | tail -1 | cut -c1-300)"
  fi
  [ -f $M/model_final.pt ] || { echo "체크포인트 없음 $n (로그 logs/prism_prem_$n.log)"; return 1; }
  [ -f $I/${S1}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $M/model_final.pt --out $I --subs $S1 --perceptive keep --tok_pool 1 | grep -E "범주|Error"
  [ -f $I/sub-0008_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $M/model_final.pt --out $I --subs $OTH --perceptive keep --save_tokens 0 --tok_pool 1 | grep -E "범주|Error"
  [ -f $O/judge/$n.json ] || $PY $PR/sep_eval.py --comp $(for s in ${SUBS//,/ }; do echo $I/${s}_keep/components.npz; done) --vt $VT --out $O/judge/$n.json --name $n | grep -E "^SEP|Error"
  $PY $PR/text_prompts.py --comp $I/${S1}_keep/components.npz --out $I/prompt_text.json > /dev/null
  [ $(cnt $I/${S1}_keep/glb_text) -ge 104 ] || trel gen --pred $I/${S1}_keep/tokens --slat_text $I/prompt_text.json --text_model $TM --out $I/${S1}_keep/glb_text | grep -E "GEN_DONE|Error|없음"
  echo "GLB $n S1 텍스트판 $(cnt $I/${S1}_keep/glb_text)개 $(date +%H:%M:%S)"
  if [ "$APAC" = 1 ]; then
    for x in AP APAC; do
      local d=$I/apac/${FS[$x]}_as01_keep
      [ $(find $DR/${FS[$x]}/npy_data -name '*.npy' 2>/dev/null | wc -l) -ge ${NN[$x]} ] && [ -f ${LS[$x]} ] || { echo "$x 데이터 없음 → 건너뜀"; continue; }
      [ -f $d/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $M/model_final.pt --out $I/apac --subs $S1 --fmri_sub ${FS[$x]} --list ${LS[$x]} --perceptive keep | grep -E "범주|Error"
      [ $(cnt $d/glb) -ge ${NN[$x]} ] || trel gen --pred $d/tokens --out $d/glb | grep -E "GEN_DONE|Error"
      echo "GLB $n $x $(cnt $d/glb)개 $(date +%H:%M:%S)"
    done
  fi
}
J=""; for n in $RUNS; do one $n & J="$J $!"; sleep 90; done
wait $J
echo "=== 요약 (판정 4: 피험자별 test 범주 정확도 = [val] 줄 끝)"
for n in $RUNS; do echo "$n | $(grep -E '^\[val\]' $W/logs/prism_prem_$n.log | tail -1 | sed 's/.*| 피험자별/피험자별/' | cut -c1-260)"; done
echo "=== PREM_DONE $(date +%H:%M:%S)"
