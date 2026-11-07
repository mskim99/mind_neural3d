#!/bin/bash
# 색 코드 실험 (11-08, Pro6000 GPU 1): Semantic 안 색 코드 (prism.py --color 1). 학습 설정은 fig와 같음 (7명, bs 14, 4만 스텝) + 색 코드
#  1 학습 prism_color  2 추론: sub-0001 none · pred · cat_sample · gt / sub-0008 none · pred · gt (sub-0008 = 원 신호에서 색이 잡히는 피험자)
#  3 빠른 확인: sub-0001 정성 14물체 × 모드 4 Gaussian 렌더 → color_quick.tar (COLOR_QUICK_DONE)
#  4 GLB 104 → 8지표 (앞 평가와 다음 GLB를 겹쳐 돌림)  5 색 일치(color_agree.py) · 렌더 색 · 정성(물체 5 × 뷰 0·2·4) · 묶기 → color.tar (COLOR_DONE)
#  비교 기준: fig = 이미 평가한 $SPP_ROOT/out/fig (색 코드 없는 같은 설정, sub-0001)
#  gt 모드는 자극 색을 넣은 상한 (방법 아님)
# 사용: nohup bash run_color.sh > $W/logs/prism_color.log 2>&1 < /dev/null & disown
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
[ -f $PR/data/stim_color.npz ] || { echo "색 정답 없음 $PR/data/stim_color.npz (git pull 확인)"; exit 1; }
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
ENC=$W/weights/MinD-3D-pp/fMRI-Shape; PY=$W/venv/bin/python
M=$W/prism_color; CK=$M/model_final.pt; I=$M/infer; O=$M/eval
RUNS="sub-0001:none sub-0001:pred sub-0001:cat_sample sub-0001:gt sub-0008:none sub-0008:pred sub-0008:gt"
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
tok() { [ $2 = none ] && echo $I/${1}_keep/tokens || echo $I/${1}_keep_$2/tokens; }
nm() { echo col_${1#sub-}_$2; }
cd $W/run

step "1 학습 (fig 설정 + --color 1, 7명 4만 스텝)"
if [ -f $CK ]; then echo "이미 있음 $CK"; else
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --init_ckpt $INIT --bs 14 --max_step 40000 \
    --per_subject 1 --enc_dir $ENC --arch fig --sem_clip front --var_heads sr --var_mode feat --color 1 --out_dir $M > $W/logs/prism_color_train.log 2>&1
fi
[ -f $CK ] || { echo "학습 실패: $W/logs/prism_color_train.log"; tail -20 $W/logs/prism_color_train.log; exit 1; }
echo "학습 끝 | $(grep -E '^\[val\]' $W/logs/prism_color_train.log | tail -1)"

step "2 추론"
for r in $RUNS; do
  s=${r%:*}; m=${r#*:}
  [ -d $(tok $s $m) ] && [ $(ls $(tok $s $m) | wc -l) -ge 104 ] && { echo "이미 있음 $s $m"; continue; }
  $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I --subs $s --perceptive keep --color_mode $m | grep -E "범주|색 머리|DONE|Error"
done

step "3 빠른 확인 (sub-0001 정성 14물체)"
for m in none pred cat_sample gt; do
  q=$O/quick/$m; [ $(cat $q/gscheck.jsonl 2>/dev/null | wc -l) -ge 14 ] && continue
  [ -d $q ] && mv $q $q.bad_$(date +%s)                                       # 끊긴 결과는 옆으로 치움 (rm 안 씀)
  trel gscheck --pred $(tok sub-0001 $m) --out $q --names $PR/scripts/qual14_uids.txt | grep -E "GSCHECK_DONE|Error|없음"
done
cd $O && tar cf color_quick.tar quick/*/*.png quick/*/gscheck.jsonl && echo "PACK quick $(ls quick/*/*.png | wc -l)개 (예상 56)"
step "COLOR_QUICK_DONE"

step "4 GLB 104 → 8지표"
EV=""
for r in $RUNS; do
  s=${r%:*}; m=${r#*:}; n=$(nm $s $m); g=$O/glb/$n
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $(tok $s $m) --out $g | grep -E "GEN_DONE|Error"
  echo "GLB $n $(ls $g/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ -n "$EV" ] && wait $EV
  (cd $SPP_ROOT && bash run_spp.sh $n $g glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_$n.log 2>&1
   echo "EVAL $n 끝 $(date +%H:%M:%S) | $(tail -2 $W/logs/spp_$n.log | head -1)") &
  EV=$!
done
wait

step "5 색 일치 · 묶기"
P=$O/pack; mkdir -p $P
CA="fig=$SPP_ROOT/out/fig/render_a_L16000"; for r in $RUNS; do CA="$CA $(nm ${r%:*} ${r#*:})=$SPP_ROOT/out/$(nm ${r%:*} ${r#*:})/render_a_L16000"; done
$PY $PR/scripts/color_agree.py --npz $PR/data/stim_color.npz --ids $DR/annotations/core_test_list.txt --train_ids $DR/annotations/core_train_list.txt \
  --json $P/color_agree.json $CA | tee $P/color_agree.txt
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for n in fig $(for r in $RUNS; do nm ${r%:*} ${r#*:}; done); do
  R=$SPP_ROOT/out/$n; mkdir -p $P/$n; cp $R/values.txt $P/$n/ 2>/dev/null || echo "지표 없음 $n"
  for u in $UIDS; do
    f=$(find $R/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
    [ -n "$f" ] && mkdir -p $P/$n/qual/$u && cp $f/{0,2,4}.png $P/$n/qual/$u/
  done
done
for r in $RUNS; do s=${r%:*}; m=${r#*:}; d=$(dirname $(tok $s $m)); cp $d/components.npz $P/$(nm $s $m)/ 2>/dev/null; done
cp $M/train_log.csv $W/logs/prism_color_train.log $P/ 2>/dev/null
cd $O && tar cf color.tar pack && echo "PACK $(find pack -name '*.png' | wc -l)개 (예상 120) $(du -sh color.tar | cut -f1)"
echo "=== 결과"; for n in fig $(for r in $RUNS; do nm ${r%:*} ${r#*:}; done); do echo "$n $(cat $P/$n/values.txt 2>/dev/null)"; done
step "COLOR_DONE"
