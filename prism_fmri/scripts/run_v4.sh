#!/bin/bash
# V-4 1단계 (11-24 사용자 확인: "진행 + 상한 1세트", Pro6000 GPU 1) — 결과 묶음 $W/prism_v4/v4_pack.tar, 로그 $W/logs/v4.log
#  ① 준비: 최종 Ours 7명 모델(prism_full/ausvp)로 피험자 1 · 6 × train · test 추론 (토큰 + V 특징 --save_var), test 외형 문장 / 정답 형상 잠재 1,403개 (v4_train.py encz)
#  ② 학습 (직렬): A = 조건 토큰만, B = 토큰 + V (문 달린 V cross-attention). cross-attention만 · bs 8 · 4,000 step (v4_train.py train)
#  ③ 형상 잠재 생성: A · B × 피험자 1 · 6 test 104 + 상한(정답 잠재 그대로, 외형 문장 = 정답 범주) → 외형(텍스트판) GLB (trellis_tools.py gen --ss_z)
#  ④ 방향 보정 · 8지표 (기준 번들 fair_eval_ref, 축 고정) — 세트 이름 v4_A_s01 · v4_A_s06 · v4_B_s01 · v4_B_s06 · v4_oracle
#  STAGE=all | post(학습 건너뜀, 이미 있는 것은 건너뜀) / STEPS · BS로 바꿀 수 있음
# 사용: GPU=1 nohup bash run_v4.sh > $W/logs/v4.log 2>&1 < /dev/null & disown   (진행: watch -n 1 bash status_v4.sh)
SD=$(cd "$(dirname "$0")" && pwd); source $SD/common.sh || exit 1
STAGE=${STAGE:-all}; V=$W/prism_v4; PY=$W/venv/bin/python; CK=$W/prism_full/ausvp/model_final.pt; STEPS=${STEPS:-4000}; BS=${BS:-8}
TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}; GENPAR=${GENPAR:-2}; PAR=${PAR:-4}; FER=$W/fair_eval_ref
QUAL="17072ff9ec47bfadc3fd2392bc923b2f 1b77f2fbf17f28e726cd2499b0c05f5a 336fcd767dadec9fb611871e114769d9 4dc9f86110ea40aaa570c6c691c987a8 5695a98b66b2b9a9c81ddeca50aa3117 6e91124a4ecaa1a365564d9c2ed3b493 969f830a531682cef155d75bbf62b80 a51410ca51a40c2529412f6cb23454eb b5d0ae4f723bce81f119374ee5d5f944"
mgpy() { (cd $MG && source $MG/mg_guard.sh > /dev/null && export HF_HUB_OFFLINE=1 && python "$@"); }
echo "=== V-4 1단계 시작 $(date '+%m-%d %H:%M') | STAGE=$STAGE | step $STEPS · bs $BS | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"
for f in $PR/v4_train.py $PR/infer_prism.py $PR/text_prompts.py $PR/trellis_tools.py $SD/run_yaw_v2.sh $CK $FER/run_eval.sh; do [ -e $f ] || { echo "없음 $f — 멈춤"; exit 1; }; done
[ "$(find $FER/gt_pc -type f | wc -l) $(find $FER/gt_render -type f | wc -l)" = "326 624" ] || { echo "평가 번들 정답 개수가 다름 — 멈춤"; exit 1; }
echo "코드 $(cd $PR && git log --oneline -1 | cut -c1-60) | 모델 $(md5sum < $CK | cut -c1-8)"

# ---- ① 준비
wait_gpu_free "V-4 추론"
for s in 01 06; do for sp in train test; do
  d=$V/infer_$sp/sub-00${s}_keep; n=$([ $sp = train ] && echo 1300 || echo 104)
  if [ $(ls $d/var 2>/dev/null | wc -l) -lt $n ]; then
    (cd $W/run && $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --subs sub-00$s --split $sp --save_var 1 --out $V/infer_$sp > $W/logs/v4_infer_${sp}_s$s.log 2>&1)
  fi
  echo "INFER|s$s $sp|토큰 $(ls $d/tokens | wc -l) · V $(ls $d/var 2>/dev/null | wc -l) | $(grep -aoE '범주 정확도 [0-9.]+' $W/logs/v4_infer_${sp}_s$s.log 2>/dev/null | tail -1)"
done; $PY $PR/text_prompts.py --comp $V/infer_test/sub-00${s}_keep/components.npz --out $V/infer_test/sub-00${s}_keep/prompt_text.json > /dev/null; done
$PY $PR/text_prompts.py --comp $V/infer_test/sub-0001_keep/components.npz --out $V/prompt_text_gt.json --gt 1 > /dev/null   # 상한: 정답 범주 문장
mgpy $PR/v4_train.py encz --tgt $TGT --out $V/zgt | grep -aE "ENCZ_DONE|Error|Traceback"
mkdir -p $V/gen/oracle/z; for u in $(cut -d/ -f2 $DR/annotations/core_test_list.txt | tr -d '\r'); do [ -f $V/zgt/$u.npy ] && cp -n $V/zgt/$u.npy $V/gen/oracle/z/; done
echo "PREP|정답 잠재 $(ls $V/zgt | wc -l) · 상한 test $(ls $V/gen/oracle/z | wc -l)"

# ---- 동작 확인: B(토큰 + V)를 20 step 학습 → 2개 형상 잠재 → 점유 확인. 실패하면 멈춤 (몇 시간 뒤에 깨지지 않게)
if [ ! -f $V/smoke/OK ]; then wait_gpu_free "V-4 동작 확인"
  mgpy $PR/v4_train.py train --cond tokv --data $V/infer_train/sub-0001_keep --z $V/zgt --out $V/smoke --steps 20 --bs $BS --save 20 > $W/logs/v4_smoke.log 2>&1
  mgpy $PR/v4_train.py genz --cond tokv --ckpt $V/smoke/ft_20.pt --data $V/infer_test/sub-0001_keep --out $V/smoke/z --limit 2 >> $W/logs/v4_smoke.log 2>&1
  if grep -q GENZ_DONE $W/logs/v4_smoke.log && [ $(ls $V/smoke/z 2>/dev/null | wc -l) -ge 2 ]; then touch $V/smoke/OK; echo "SMOKE OK | $(grep -aE '^\[train tokv\]' $W/logs/v4_smoke.log | tail -1)"
  else echo "SMOKE 실패 — 멈춤 ($W/logs/v4_smoke.log)"; grep -aE "Error|Traceback|line [0-9]+" $W/logs/v4_smoke.log | tail -4; exit 1; fi; fi

# ---- ② 학습 (직렬)
DATA=$V/infer_train/sub-0001_keep,$V/infer_train/sub-0006_keep
if [ $STAGE = all ]; then for r in A:tok B:tokv; do n=${r%%:*} c=${r#*:}
  [ -f $V/ft/$n/ft_$STEPS.pt ] && { echo "이미 있음 $n"; continue; }
  wait_gpu_free "V-4 학습 $n"; echo "=== 학습 $n ($c) $(date +%H:%M:%S)"
  mgpy $PR/v4_train.py train --cond $c --data $DATA --z $V/zgt --out $V/ft/$n --steps $STEPS --bs $BS --save $((STEPS / 2)),$STEPS > $W/logs/v4_train_$n.log 2>&1
  echo "TRAIN|$n|$(grep -aE '^\[train' $W/logs/v4_train_$n.log | tail -1)"; grep -aE "Error|Traceback" $W/logs/v4_train_$n.log | tail -2
done; fi

# ---- ③ 형상 잠재 → GLB
for r in A:tok B:tokv; do n=${r%%:*} c=${r#*:}; [ -f $V/ft/$n/ft_$STEPS.pt ] || { echo "체크포인트 없음 $n — 건너뜀"; continue; }
  for s in 01 06; do wait_gpu_free "V-4 형상 $n s$s"
    mgpy $PR/v4_train.py genz --cond $c --ckpt $V/ft/$n/ft_$STEPS.pt --data $V/infer_test/sub-00${s}_keep --out $V/gen/${n}_s$s/z | grep -aE "GENZ_DONE|Error|Traceback"
  done; done
glb_one() { local g=$1 pt=$2 d=$V/gen/$1; [ -d $d/z ] || return; local k=$(ls $d/z | wc -l)
  [ $(ls $d/glb/*/*.glb 2>/dev/null | wc -l) -ge $k ] || { wait_gpu_free "GLB $g"; mgpy $PR/trellis_tools.py gen --ss_z $d/z --slat_text $pt --text_model $TM --out $d/glb 2>&1 | grep -aE "GEN_DONE|Error|없음"; }
  ls $d/glb/*/*.glb 2>/dev/null | wc -l > $d/NEED; echo "GLB|v4_$g|$(cat $d/NEED)/$k|$(date +%H:%M:%S)"; }
SETL="A_s01 A_s06 B_s01 B_s06 oracle"
ptf() { case $1 in oracle) echo $V/prompt_text_gt.json;; *) echo $V/infer_test/sub-00${1##*_s}_keep/prompt_text.json;; esac; }
tkd() { case $1 in oracle) echo $V/infer_test/sub-0001_keep/tokens;; *) echo $V/infer_test/sub-00${1##*_s}_keep/tokens;; esac; }
ES=""; for g in $SETL; do [ -d $V/gen/$g/z ] && ES="$ES v4_$g:$V/gen/$g/glb:$(tkd $g):full:v2"; done
SETS="$ES" PAR=$PAR WAIT=1 FE2=$FER FE_RUN=run_eval.sh GPU=$CUDA_VISIBLE_DEVICES bash $SD/run_yaw_v2.sh > $W/logs/v4_yaw.log 2>&1 & EV=$!
lane() { local i=0; for g in $SETL; do [ $(( i % GENPAR )) = $1 ] && glb_one $g $(ptf $g); i=$(( i + 1 )); done; }
J=""; for l in $(seq 0 $(( GENPAR - 1 ))); do lane $l & J="$J $!"; sleep 5; done; wait $J
echo "=== GLB 끝 $(date +%H:%M:%S)"; wait $EV

# ---- ⑤ 요약 · 묶음
echo "=== 요약"; grep -aE "^YV2\|v4_" $W/logs/v4_yaw.log
F=""; for g in $SETL; do O=$FER/out/yaw_v4_$g; r=$(find $O/render_R2 -name '*.png' 2>/dev/null | wc -l); echo "RENDER|v4_$g|$r/624$([ $r -lt 624 ] && echo ' 부족')"
  for f in values.txt frame.txt logs/frame.log; do [ -f $O/$f ] && F="$F ${O#$W/}/$f"; done
  for u in $QUAL; do [ -d $O/render_R2/$u ] && F="$F ${O#$W/}/render_R2/$u"; done
  for u in $QUAL; do ls $V/gen/$g/glb/$u/*_snap.png > /dev/null 2>&1 && F="$F prism_v4/gen/$g/glb/$u/${u}_snap.png"; done; done
F="$F $(cd $W && ls -d prism_v4/ft/*/train_log.csv prism_v4/infer_test/*/components.npz prism_v4/infer_test/*/prompt_text.json prism_v4/prompt_text_gt.json logs/v4*.log 2>/dev/null | tr '\n' ' ')"
(cd $W && tar cf $V/v4_pack.tar $F) && echo "PACK $V/v4_pack.tar $(tar tf $V/v4_pack.tar | wc -l)개 · $(du -h $V/v4_pack.tar | cut -f1)"
echo "=== V4_DONE $(date +%H:%M:%S)"
