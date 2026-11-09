#!/bin/bash
# B안 1단계 후반 (11-10, 5090): 렌더가 다 끝나면 → DINO 토큰 빈 곳 채우기 → 조건 없는 prior (PSTEPS) → 생성 64 · 복사 검사 → 생성 16개 렌더 · 몽타주
#  run_b_all.sh는 처음 렌더 3개만 기다려서 토큰이 일부 빠짐 + 기본 6만 스텝은 아침까지 못 끝나 → 이 스크립트로 이어 감
# 사용: GPU=1 PSTEPS=30000 nohup bash run_b_prior.sh > $W/logs/prism_b_prior2.log 2>&1 < /dev/null & disown
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_b; P=$O/prior_u
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run
step "0 렌더 끝 기다림"
while pgrep -f "[r]ender_b.py" >/dev/null; do sleep 30; done
echo "렌더 $(ls $O/render/*/0.png | wc -l)개 (목록 10010)"
step "1 DINO 토큰 빈 곳 채우기"
trel dino --frames $O/render --out $O/tok | grep -E "DINO_DONE|Error"
echo "토큰 $(ls $O/tok/*.npy | wc -l)개, $(du -sh $O/tok | cut -f1)"
step "2 조건 없는 prior (${PSTEPS:-30000} 스텝, bs 32)"
[ -f $P/prior_final.pt ] || $PY $PR/prior_token.py train --uncond_dir $O/tok --out $P --steps ${PSTEPS:-30000} --bs 32 --save_every 5000 > $P.log 2>&1
grep -E "\[data\]|\[model\]|^\[val\]" $P.log | tail -5
step "3 생성 64 · 복사 검사"
$PY $PR/prior_token.py sample --uncond_n 64 --ckpt $P/prior_final.pt --out $O/gen_u | grep -E "생성|COPY|Error"
mkdir -p $O/gen_u16; ls $O/gen_u/gen_00*.npy | head -16 | xargs -I{} ln -sf {} $O/gen_u16/
step "4 생성 16개 렌더 (형상 · 외형 둘 다 생성 토큰)"
trel gscheck --pred $O/gen_u16 --out $O/gen_u_render | grep -E "GSCHECK_DONE|Error"
echo "  생성 색: $($PY $PR/scripts/gs_summary.py $O/gen_u_render/gscheck.jsonl)"
$PY - <<PYEOF
import os, glob, json
from PIL import Image, ImageDraw
O='$O'; S=200
gen=sorted(glob.glob(O+'/gen_u_render/*_gs.png'))[:16]; cc=json.load(open(O+'/gen_u/copy_check.json'))
c=Image.new('RGB',(S*8,40+S*2*2),'white'); d=ImageDraw.Draw(c); d.text((4,4),'B prior samples (unconditional) - TRELLIS Gaussian render, first view; number = max cos to train tokens',fill='black')
for i,f in enumerate(gen):
    im=Image.open(f).convert('RGB'); im=im.crop((0,0,im.height,im.height)).resize((S,S)); r,col=divmod(i,8); c.paste(im,(col*S,40+r*S))
    d.text((col*S+4,40+r*S+4),'%.2f'%cc['gen'][int(os.path.basename(f)[4:8])],fill='blue')
real=sorted(glob.glob(O+'/render/*/0.png'))[::700][:16]
for i,f in enumerate(real):
    r,col=divmod(i,8); c.paste(Image.open(f).convert('RGB').resize((S,S)),(col*S,40+(2+r)*S))
d.text((4,40+2*S+4),'real renders (training data examples)',fill='black')
c.save(O+'/gen_u_비교.png'); print('MONTAGE', O+'/gen_u_비교.png')
PYEOF
step "B_PRIOR_DONE"
