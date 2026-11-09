#!/bin/bash
# V = 범주 안 개체 형상 차이 (11-09, Pro6000 GPU 1): 전제 확인 결과 — 시점 변화는 fMRI에 거의 없고(실루엣 · DINO 모두 r ≤ 0.15), 범주 넘는 고정 형상은 뚜렷(폭 r 0.51~0.66, 7명)
#  정답: 물체 평균 실루엣 [cy, w, h, area] − train 범주 평균 (build_view_targets.py 'shape', train 통계만) → V가 프레임마다 같은 값을 맞춤
#  1 정답 → $TGT/view_targets_shape.npz  2 학습 vshape_g0 · vshape_g1 동시 (fig 설정 7명 bs 14 4만 스텝, V 범주 GRL λ 0 / $LAM)
#  3 평가 (학습 끝나면): 분리 프로브(probe_split.py: V · S · P로 범주 · 크기, fig도) + sub-0001 test 추론 → GLB → 8지표(이전 설정, 임시)
#  → vshape.tar (VSHAPE_DONE)
# 사용: LAM=0.5 nohup bash run_vshape.sh > $W/logs/prism_vshape.log 2>&1 < /dev/null & disown
SUBS=sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008
source $(dirname $0)/common.sh || exit 1
export SPP_ROOT=$W/spp_eval SPP_PY=$W/venv/bin/python SPP_BLENDER=$W/tools/blender-4.2.3-linux-x64/blender
ENC=$W/weights/MinD-3D-pp/fMRI-Shape; PY=$W/venv/bin/python; S1=sub-0001; O=$W/prism_vshape; LAM=${LAM:-0.5}; VT=$TGT/view_targets_shape.npz
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $O
[ -f $TGT/view_sil.npz ] || { echo "입력 없음: $TGT/view_sil.npz"; exit 1; }

step "1 정답 (범주 안 개체 형상 차이)"
$PY $PR/build_view_targets.py --sil $TGT/view_sil.npz --data_root $DR --out $VT | grep -E "shape|DONE|Error"

train() {   # 이름 · λ
  local n=$1 l=$2; [ -f $O/$n/model_final.pt ] && { echo "이미 있음 $n"; return; }
  $PY $PR/train_prism.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs $SUBS --init_ckpt $INIT --bs 14 --max_step 40000 \
    --per_subject 1 --enc_dir $ENC --arch fig --sem_clip front --var_mode feat --var_heads shape --view_tgt $VT --lam_vadv $l --resume \
    --out_dir $O/$n > $W/logs/prism_vs_$n.log 2>&1
}
probe() {   # 이름 · 체크포인트
  [ -f $O/probe/$1/probe_split.json ] || $PY $PR/probe_split.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --ckpt $2 --out $O/probe/$1 --subs $S1 > $W/logs/prism_vs_probe_$1.log 2>&1
  echo "PROBE $1 | $(grep -E '^PROBE' $W/logs/prism_vs_probe_$1.log | grep -vE '기준선' | cut -c1-200 | tr '\n' ' ')"
}
evaluate() {   # 이름
  local n=$1 ck=$O/$1/model_final.pt I=$O/infer/$1
  [ -f $ck ] || { echo "체크포인트 없음 $n"; return; }
  probe $n $ck
  [ -f $I/${S1}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $ck --out $I --subs $S1 --perceptive keep | grep -E "범주|DONE|Error"
  [ $(ls $I/${S1}_keep/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/${S1}_keep/tokens --out $I/${S1}_keep/glb | grep -E "GEN_DONE|Error"
  echo "GLB $n $(ls $I/${S1}_keep/glb/*/*.glb 2>/dev/null | wc -l)개"
  (cd $SPP_ROOT && bash run_spp.sh vs_$n $I/${S1}_keep/glb glb $CUDA_VISIBLE_DEVICES > $W/logs/spp_vs_$n.log 2>&1)
  echo "EVAL $n | $(grep -E '^2way' $W/logs/spp_vs_$n.log | tail -1)"
}

step "2 학습: V = 범주 안 개체 형상 차이 (GRL 0 · $LAM 동시)"
train vshape_g0 0 & T1=$!; sleep 90; train vshape_g1 $LAM & T2=$!
probe fig $W/prism_fig/model_final.pt
wait $T1 $T2
for n in vshape_g0 vshape_g1; do echo "학습 끝 $n | $(grep -E '^\[val\]' $W/logs/prism_vs_$n.log | tail -1 | cut -c1-260)"; done

step "3 평가"
for n in vshape_g0 vshape_g1; do evaluate $n; done

step "4 묶기"
P=$O/pack; mkdir -p $P
UIDS="6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944"
for n in vshape_g0 vshape_g1; do
  mkdir -p $P/$n; cp $O/$n/train_log.csv $W/logs/prism_vs_$n.log $P/$n/ 2>/dev/null; cp $SPP_ROOT/out/vs_$n/values.txt $P/$n/ 2>/dev/null
  cp $O/infer/$n/${S1}_keep/components.npz $P/$n/ 2>/dev/null
  for u in $UIDS; do f=$(find $SPP_ROOT/out/vs_$n/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1); [ -n "$f" ] && mkdir -p $P/$n/qual/$u && cp $f/{0,2,4}.png $P/$n/qual/$u/; done
done
cp -r $O/probe $P/ 2>/dev/null; cp $W/prism_vchange/probe_view_dino.json $P/ 2>/dev/null
cd $O && tar cf vshape.tar pack && echo "PACK $(find pack -name '*.png' | wc -l)개 (예상 30)"
echo "=== 결과"; for n in vshape_g0 vshape_g1; do echo "$n $(cat $SPP_ROOT/out/vs_$n/values.txt 2>/dev/null)"; done
step "VSHAPE_DONE"
