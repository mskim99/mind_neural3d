#!/bin/bash
# V = "변하는 정보" 실험 (11-09, Pro6000 GPU 1): 사용자 결정 — V 정답을 실루엣 / 시점 특징의 변하는 부분으로, V 범주 GRL 유무 → 4개 학습 · 비교
#  입력(로컬에서 올림): $TGT/view_sil.npz, $TGT/vframes/<uid>/{2..7}.png (prep_view_targets.py)
#  1 (뒤에서) 시점 특징: trellis_tools.py dino --pool 1, fMRI 프레임 2~7 → $TGT/dinof_v{2..7} (약 4시간)
#  2 실루엣 정답 → $TGT/view_targets.npz (build_view_targets.py, train 통계만)
#  3 학습 vsil_g0 · vsil_g1 동시 (fig 설정 7명 bs 14 4만 스텝, V = 실루엣 변화, GRL λ 0 / $LAM)
#  4 시점 특징 끝나면 정답 다시(sil + view) → 학습 vview_g0 · vview_g1 동시
#  5 평가(학습 끝나는 대로, 뒤에서): sub-0001 test 추론 → GLB → 8지표(이전 설정, 임시) + 분리 프로브(probe_split.py: V로 범주 · 변하는 정보)
#    비교 기준 fig(V = Scale · Ratio)도 분리 프로브 한 번
#  → vchange.tar (VCHANGE_DONE)
# 사용: LAM=0.5 nohup bash run_vchange.sh > $W/logs/prism_vchange.log 2>&1 < /dev/null & disown
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
ENC=$W/weights/MinD-3D-pp/fMRI-Shape; PY=$W/venv/bin/python; S1=sub-0001; O=$W/prism_vchange; LAM=${LAM:-0.5}; VT=$TGT/view_targets.npz
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $O
[ -f $TGT/view_sil.npz ] && [ $(ls -d $TGT/vframes/*/ 2>/dev/null | wc -l) -ge 1400 ] || { echo "입력 없음: $TGT/view_sil.npz · $TGT/vframes (로컬에서 올리기)"; exit 1; }

step "1 시점 특징 DINO 요약 (뒤에서, 로그 logs/prism_dinof.log)"
( for vw in 2 3 4 5 6 7; do
    [ $(ls $TGT/dinof_v$vw/*.npy 2>/dev/null | wc -l) -ge 1404 ] || trel dino --frames $TGT/vframes --out $TGT/dinof_v$vw --view $vw --pool 1 2>&1 | grep -E "DINO_DONE|Error|error|cond shape"
    echo "dinof_v$vw $(ls $TGT/dinof_v$vw/*.npy 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  done ) > $W/logs/prism_dinof.log 2>&1 &
DV=$!

step "2 실루엣 정답"
$PY $PR/build_view_targets.py --sil $TGT/view_sil.npz --data_root $DR --out $VT | grep -E "sil|DONE|Error"

train() {   # 이름 · var_heads · λ
  local n=$1 h=$2 l=$3; [ -f $O/$n/model_final.pt ] && { echo "이미 있음 $n"; return; }
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --init_ckpt $INIT --bs 14 --max_step 40000 \
    --per_subject 1 --enc_dir $ENC --arch fig --sem_clip front --var_mode feat --var_heads $h --view_tgt $VT --lam_vadv $l --resume \
    --out_dir $O/$n > $W/logs/prism_vc_$n.log 2>&1
}
evaluate() {   # 이름 · 체크포인트
  local n=$1 ck=$2 I=$O/infer/$1
  [ -f $ck ] || { echo "체크포인트 없음 $n"; return; }
  $PY $PR/probe_split.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --ckpt $ck --out $O/probe/$n --subs $S1 > $W/logs/prism_vc_probe_$n.log 2>&1
  echo "PROBE $n | $(grep -E '^PROBE (var|sem|perc)' $W/logs/prism_vc_probe_$n.log | head -3 | tr '\n' ' ')"
  [ "$n" = fig ] && return
  [ -f $I/${S1}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $ck --out $I --subs $S1 --perceptive keep | grep -E "범주|DONE|Error"
  [ $(ls $I/${S1}_keep/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/${S1}_keep/tokens --out $I/${S1}_keep/glb | grep -E "GEN_DONE|Error"
  (cd $SPP_ROOT && bash run_spp.sh vc_$n $I/${S1}_keep/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_vc_$n.log 2>&1)
  echo "EVAL $n | $(grep -E '^2way' $W/logs/spp_vc_$n.log | tail -1)"
}

step "3 학습: V = 실루엣 변화 (GRL 0 · $LAM 동시)"
train vsil_g0 sil 0 & T1=$!; sleep 90; train vsil_g1 sil $LAM & T2=$!
evaluate fig $W/prism_fig/model_final.pt
wait $T1 $T2
for n in vsil_g0 vsil_g1; do echo "학습 끝 $n | $(grep -E '^\[val\]' $W/logs/prism_vc_$n.log | tail -1)"; done
( for n in vsil_g0 vsil_g1; do evaluate $n $O/$n/model_final.pt; done ) > $W/logs/prism_vc_eval_sil.log 2>&1 &
E1=$!

step "4 시점 특징 기다림 → 정답 다시 → 학습: V = 시점 특징 변화"
wait $DV; grep -E "^dinof_v|Error" $W/logs/prism_dinof.log
for vw in 2 3 4 5 6 7; do [ $(ls $TGT/dinof_v$vw/*.npy 2>/dev/null | wc -l) -ge 1400 ] || { echo "dinof_v$vw 부족 → logs/prism_dinof.log"; exit 1; }; done
$PY $PR/build_view_targets.py --sil $TGT/view_sil.npz --data_root $DR --out $VT --dino $TGT --pc 32 | grep -E "sil|view|DONE|Error"
train vview_g0 view 0 & T3=$!; sleep 90; train vview_g1 view $LAM & T4=$!
wait $T3 $T4
for n in vview_g0 vview_g1; do echo "학습 끝 $n | $(grep -E '^\[val\]' $W/logs/prism_vc_$n.log | tail -1)"; done
wait $E1; cat $W/logs/prism_vc_eval_sil.log
for n in vview_g0 vview_g1; do evaluate $n $O/$n/model_final.pt; done

step "5 묶기"
P=$O/pack; mkdir -p $P
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for n in vsil_g0 vsil_g1 vview_g0 vview_g1; do
  mkdir -p $P/$n; cp $O/$n/train_log.csv $P/$n/ 2>/dev/null; cp $SPP_ROOT/out/vc_$n/values.txt $P/$n/ 2>/dev/null
  for u in $UIDS; do f=$(find $SPP_ROOT/out/vc_$n/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1); [ -n "$f" ] && mkdir -p $P/$n/qual/$u && cp $f/{0,2,4}.png $P/$n/qual/$u/; done
done
cp -r $O/probe $P/ 2>/dev/null; cp $VT $P/ 2>/dev/null
cd $O && tar cf vchange.tar pack && echo "PACK $(find pack -name '*.png' | wc -l)개 (예상 60)"
echo "=== 결과"; for n in vsil_g0 vsil_g1 vview_g0 vview_g1; do echo "$n $(cat $SPP_ROOT/out/vc_$n/values.txt 2>/dev/null)"; done
step "VCHANGE_DONE"
