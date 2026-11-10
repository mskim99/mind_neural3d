#!/bin/bash
# B안 2단계 (11-10, 5090): 토큰 압축 → 복원만으로 TRELLIS 렌더가 유지되는지 (학습 없음)
#  전역 PCA 2개 (c64_s2 · c128_s2) → 검증 8물체 cfg별 복원 → gscheck 렌더 (GPU 0 · 1 나눔) → 몽타주
# 사용: nohup bash run_b_comp.sh > $W/logs/prism_b_comp.log 2>&1 < /dev/null & disown
GPU=0 source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_b; C=$O/comp; TC="$PY $PR/tok_compress.py"
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run
step "1 전역 PCA"
for c in c64_s2 c128_s2; do [ -f $C/glob_$c.pt ] || $TC gfit --tok $O/tok --objs $O/prior_u/train_objs.txt --out $C --cfg $c --kmax 2048 | grep -E "전역|Error"; done
step "2 복원 (검증 8물체)"
CF=c32_s1,c64_s1,c128_s1,c64_s2,c128_s2,c256_s2,c64_s2_g512,c64_s2_g2048,c128_s2_g2048
$TC recon --tok $O/tok --objs $O/prior_u/val_objs.txt --out $C --cfgs $CF --n 8 | grep -E "recon|Error"
step "3 렌더 (원 토큰 + cfg 9개)"
L0="orig c32_s1 c64_s1 c128_s1 c64_s2 c128_s2"; L1="c256_s2 c64_s2_g512 c64_s2_g2048 c128_s2_g2048"
rend() { for c in $2; do src=$C/recon/$c; [ $c = orig ] && src=$O/tok
  CUDA_VISIBLE_DEVICES=$1 trel gscheck --pred $src --names $C/recon/names.txt --out $C/render/$c | grep -E "GSCHECK_DONE|Error"; done; }
rend 0 "$L0" & rend 1 "$L1" & wait
step "4 몽타주"
$PY - <<PYEOF
import json
from PIL import Image, ImageDraw
C='$C'; O='$O'; S=170
cf=['orig']+'$CF'.split(','); M=json.load(open(C+'/recon/metrics.json'))
names=[l.strip() for l in open(C+'/recon/names.txt') if l.strip()]
c=Image.new('RGB',(S*(len(cf)+1),60+S*len(names)),'white'); d=ImageDraw.Draw(c)
d.text((4,2),'B compression check: render of reconstructed tokens (no prior). header = cfg / latent dim / mean-token cos',fill='black')
d.text((4,40),'input render',fill='black')
for k,cc in enumerate(cf):
    m=M.get(cc,{}); d.text(((k+1)*S+4,20),cc,fill='black')
    if m: d.text(((k+1)*S+4,40),'%d / %.3f'%(m['latent_dim'],m['rep_cos']),fill='blue')
for r,u in enumerate(names):
    c.paste(Image.open(f'{O}/render/{u}/0.png').convert('RGB').resize((S,S)),(0,60+r*S))
    for k,cc in enumerate(cf):
        try: im=Image.open(f'{C}/render/{cc}/{u}_gs.png').convert('RGB'); im=im.crop((0,0,im.height,im.height)).resize((S,S)); c.paste(im,((k+1)*S,60+r*S))
        except Exception as e: pass
c.save(C+'/압축복원_비교.png'); print('MONTAGE', C+'/압축복원_비교.png')
PYEOF
step "B_COMP_DONE"
