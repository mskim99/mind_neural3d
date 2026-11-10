#!/bin/bash
# 풍부한 텍스트 조건 진단 (11-10, 사용자 승인, Pro6000 GPU 1, 학습 없음) — 설계 `.보고 자료/261112_9차보고/PRISM_분리/분리·텍스트_설계.md` 3절
#  형상(SS) = vshape_g0 예측 토큰 그대로, 외형(SLat) = TRELLIS 텍스트판 문장만 바꿈 → 104 GLB → 공정평가 8지표
#  순서가 판단 순서:
#    ① oracle_* : test 물체 자기 캡션 (상한, 방법 아님) — "a {범주}"(txt_base)보다 안 나으면 풍부한 문장 방향은 접음
#    ② retr_*   : 예측 범주(S) 안에서 V 예측 형상이 가장 가까운 train 물체의 캡션 / retr_rand_*: 같은 범주 무작위 (대조)
#  캡션 = Qwen2-VL-7B가 자극 영상 프레임 0 · 120 · 240°로 쓴 모양 · 색 · 재질 (caption_vlm.py, venv_unclip). 처음 실행 때 모델 받음(약 16 GB, $W/cache/hf)
#  → $W/prism_caption/caption.tar (CAPTION_DONE). 지표 $FE/out/cap_<이름>/values.txt
# 사용: nohup bash run_caption.sh > $W/logs/prism_caption.log 2>&1 < /dev/null & disown      (VARS="oracle_color oracle_nocolor"처럼 고를 수 있음)
source $(dirname $0)/common.sh || exit 1
VARS=${VARS:-"oracle_color oracle_nocolor retr_nocolor retr_rand_nocolor retr_color"}
FE=${FE:-$W/fair_eval}; PY=$W/venv/bin/python; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}
CK=$W/prism_vshape/vshape_g0/model_final.pt; VT=$TGT/view_targets_shape.npz; O=$W/prism_caption; I=$O/infer/sub-0001_keep
BASE=$W/prism_vshape/text/glb/vt_s1      # 이미 있는 vshape_g0 + "a {예측 범주}" GLB (run_vshape_text.sh)
export FE_PY=$PY FE_WORKERS=${FE_WORKERS:-6} TORCH_HOME=$W/cache/torch FE_BLENDER=${FE_BLENDER:-$W/tools/blender-4.2.3-linux-x64/blender}
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
step() { echo "=== $1 $(date +%H:%M:%S)"; }
# 캡션 묶음 (11-11): 4.46.3 시각 인코더는 묶음 안 이미지 전체에 attention 행렬을 한 번에 만듦 → bs 8(이미지 24장)이면 36 GiB OOM, 기본 bs 2
# 캡션 파이썬 (11-11): Pro6000 venv는 transformers 4.34.1이라 Qwen2-VL 없음 → venv_unclip이 있으면 그것, 없으면 venv + $W/pylib_cap(transformers 4.46.3만 --target 설치)
if [ -x $W/venv_unclip/bin/python ]; then CAPRUN="$W/venv_unclip/bin/python"
elif [ -d $W/pylib_cap/transformers ]; then CAPRUN="env PYTHONPATH=$W/pylib_cap $PY"
else echo "캡션 파이썬 없음: $W/venv_unclip 또는 $W/pylib_cap"; exit 1; fi
[ -f $CK ] && [ -f $VT ] && [ -f $FE/run_temp.sh ] || { echo "없음: $CK · $VT · $FE/run_temp.sh 중 하나"; exit 1; }
cd $W/run; mkdir -p $O

step "1 캡션 (train + test, 자극 프레임)"
HF_HOME=$W/cache/hf $CAPRUN $PR/caption_vlm.py --frames $TGT/frames --list $DR/annotations/core_train_list.txt $DR/annotations/core_test_list.txt \
  --out $O/captions.jsonl --bs ${CAP_BS:-2} 2>&1 | grep --line-buffered -E "^\[cap\]|CAPTION_DONE|Error|error|Traceback|File "
grep -q . $O/captions.jsonl 2>/dev/null || { echo "캡션 없음 → 멈춤"; exit 1; }

step "2 vshape_g0 sub-0001 추론 (components에 vt · clip) → 문장"
[ -f $I/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $O/infer --subs sub-0001 --perceptive keep | grep -E "범주|Error"
$PY $PR/caption_prompts.py --cap $O/captions.jsonl --comp $I/components.npz --vt $VT --train_list $DR/annotations/core_train_list.txt --out $O/prompts

step "3 GLB → 8지표 (기준 txt_base 먼저, 지표는 한 번에 하나씩 뒤에서)"
EV=""
ev() {   # 이름 · GLB 폴더
  [ -n "$EV" ] && wait $EV
  ( [ -f $FE/out/cap_$1/values.txt ] || bash $FE/run_temp.sh cap_$1 $2 glb $CUDA_VISIBLE_DEVICES > $W/logs/fe_cap_$1.log 2>&1
    echo "EVAL_DONE cap_$1 $(date +%H:%M:%S) | $(grep -E '^R2\|(2way|10way|LPIPS)|^FPD\s|^P2\|CD\s' $FE/out/cap_$1/values.txt 2>/dev/null | tr '\n' ' ')" ) &
  EV=$!
}
[ $(ls $BASE/*/*.glb 2>/dev/null | wc -l) -ge 104 ] && ev txt_base $BASE || echo "기준 GLB 없음 $BASE (txt_base 건너뜀)"
for v in $VARS; do
  g=$O/glb/$v
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/tokens --slat_text $O/prompts/$v.json --text_model $TM --out $g | grep -E "텍스트판|GEN_DONE|Error|없음"
  echo "GLB $v $(ls $g/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ $(ls $g/*/*.glb 2>/dev/null | wc -l) -ge 104 ] && ev $v $g || echo "GLB $v 104개 미만 → 지표 건너뜀"
done
[ -n "$EV" ] && wait $EV

step "4 묶기"
P=$O/pack; mkdir -p $P; cp $O/captions.jsonl $P/; cp -r $O/prompts $P/
for v in txt_base $VARS; do cp $FE/out/cap_$v/values.txt $P/values_$v.txt 2>/dev/null; done
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for v in txt_base $VARS; do for u in $UIDS; do
  f=$(find $FE/out/cap_$v/render_R2 -maxdepth 3 -type d -name "${u}*" 2>/dev/null | head -1)
  [ -n "$f" ] && mkdir -p $P/qual/$v/$u && cp $f/{0,2,4}.png $P/qual/$v/$u/ 2>/dev/null
done; done
cd $O && tar cf caption.tar pack && echo "PACK $(tar tf caption.tar | wc -l)개"
echo "=== 결과"; for v in txt_base $VARS; do echo "$v | $(grep -E '^R2\|(2way|10way|LPIPS|PSNR|SSIM)|^FPD\s|^P2\|(CD|EMD)\s' $FE/out/cap_$v/values.txt 2>/dev/null | tr '\n' ' ')"; done
step "CAPTION_DONE"
