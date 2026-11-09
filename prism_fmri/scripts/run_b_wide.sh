#!/bin/bash
# B안 용량 시험 (11-10, 5090 GPU 1): 같은 데이터 · 조건 없음, DiT 폭 768 → 1152 (heads 16), 2만 5천 스텝 → 복사 검사(T 1.0 · 0.8) · 16개 렌더
#  가설(추정): 2×2 패치 묶음 4096차원을 폭 768로 처리하는 출력 병목이 무지개 얼룩 원인
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_b; P=$O/prior_w; cd $W/run
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
[ -f $P/prior_final.pt ] || $PY $PR/prior_token.py train --uncond_dir $O/tok --out $P --steps ${PSTEPS:-25000} --bs 32 --dim 1152 --heads 16 --depth 12 --save_every 5000 > $P.log 2>&1
grep -E "\[model\]|^\[val\]" $P.log | tail -3
for T in 1.0 0.8; do
  G=$O/gen_w_t$T; echo "== T $T"
  $PY $PR/prior_token.py sample --uncond_n 64 --temp $T --ckpt $P/prior_final.pt --out $G | grep -E "COPY|Error"
  mkdir -p ${G}16; ls $G/gen_00*.npy | head -16 | xargs -I{} ln -sf {} ${G}16/
  trel gscheck --pred ${G}16 --out ${G}_render | grep -E "GSCHECK_DONE|Error"
done
$PY - <<PYEOF
import glob, json, os
from PIL import Image, ImageDraw
O='$O'; S=170; rows=[('w768 T1',O+'/gen_u'),('w768 T0.8',O+'/gen_u_t0.8'),('w1152 T1',O+'/gen_w_t1.0'),('w1152 T0.8',O+'/gen_w_t0.8')]
c=Image.new('RGB',(S*16,30+S*len(rows)),'white'); d=ImageDraw.Draw(c); d.text((4,4),'B prior width test (number = max cos to train tokens; real val renders ~0.876)',fill='black')
for r,(h,G) in enumerate(rows):
    cc=json.load(open(G+'/copy_check.json'))['gen']
    for i,f in enumerate(sorted(glob.glob(G+'_render/*_gs.png'))[:16]):
        im=Image.open(f).convert('RGB'); im=im.crop((0,0,im.height,im.height)).resize((S,S)); c.paste(im,(i*S,30+r*S))
        d.text((i*S+3,30+r*S+3),'%s %.2f'%(h if i==0 else '',cc[int(os.path.basename(f)[4:8])]),fill='blue')
c.save(O+'/gen_wide_비교.png'); print('MONTAGE')
PYEOF
echo B_WIDE_DONE
