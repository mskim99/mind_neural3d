#!/bin/bash
# B안 1단계 전체 (11-10, 5090 GPU 1): ShapeNet 10,010개 렌더(자극 프레임 0 시점) → TRELLIS DINO 토큰 → 조건 없는 토큰 prior → 생성 · 복사 검사 · 렌더
#  렌더: render_b.py yaw 0 · fov 56 (실루엣 IoU 0.894), 실제 텍스처, 투명 배경 → 흰 합성, 월드 세기 LIGHT · 해 SUN (calib_look_b.py로 고름)
#  사용자 11-09: 13범주 × 770 (train 1,299 포함, test 104 제외), 프레임 0 시점 1장
# 사용: GPU=1 LIGHT=0.35 SUN=0 nohup bash run_b_all.sh > $W/logs/prism_b_all.log 2>&1 < /dev/null & disown
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; BL=$W/tools/blender-4.2.3-linux-x64/blender; SN=$W/data/shapenet_b; O=$W/prism_b; NS=${SHARDS:-3}
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $O

step "1 렌더 (shard $NS개, LIGHT ${LIGHT} SUN ${SUN})"
for k in $(seq 0 $((NS - 1))); do
  $BL -b -P $PR/render_b.py -- --mode render --root $SN --list $SN/objects_b.txt --out $O/render --fov 56 --yaw 0 --light ${LIGHT} --sun ${SUN} --shard $k/$NS \
    > $W/logs/prism_b_render_$k.log 2>&1 &
done
wait
echo "렌더 $(ls $O/render/*/0.png 2>/dev/null | wc -l)개 (예상 10010) | 실패 $(cat $W/logs/prism_b_render_*.log | grep -c LOAD_FAIL)"

step "2 DINO 토큰 (TRELLIS 전처리, 흰 배경 문턱값 — 자극과 같은 경로)"
trel dino --frames $O/render --out $O/tok | grep -E "DINO_DONE|Error|\[dino"
echo "토큰 $(ls $O/tok/*.npy | wc -l)개, $(du -sh $O/tok | cut -f1)"

step "3 조건 없는 prior 학습 (DiT 768 × 12, bs 32, ${PSTEPS:-60000} 스텝)"
[ -f $O/prior/prior_final.pt ] || $PY $PR/prior_token.py train --uncond_dir $O/tok --out $O/prior --steps ${PSTEPS:-60000} --bs 32 > $W/logs/prism_b_prior.log 2>&1
grep -E "^\[val\]|\[data\]|\[model\]" $W/logs/prism_b_prior.log | tail -4

step "4 생성 64개 · 복사 검사 · 렌더 16개"
$PY $PR/prior_token.py sample --uncond_n 64 --ckpt $O/prior/prior_final.pt --out $O/gen | grep -E "생성|COPY|Error"
mkdir -p $O/gen16; ls $O/gen/gen_00*.npy | head -16 | xargs -I{} ln -sf {} $O/gen16/
trel gscheck --pred $O/gen16 --out $O/gen_render | grep -E "GSCHECK_DONE|Error"
echo "  생성 색: $($PY $PR/scripts/gs_summary.py $O/gen_render/gscheck.jsonl)"
step "B_ALL_DONE"
