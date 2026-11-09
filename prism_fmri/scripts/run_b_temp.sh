#!/bin/bash
# B안 낮은 온도 샘플링 (11-10, 5090 GPU 1, 학습 없음): 시작 노이즈 × T (0.8 · 0.6) → 복사 검사 · 16개 렌더
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_b; P=$O/prior_u; cd $W/run
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
for T in 0.8 0.6; do
  G=$O/gen_u_t$T; echo "== T $T"
  $PY $PR/prior_token.py sample --uncond_n 64 --temp $T --ckpt $P/prior_final.pt --out $G | grep -E "COPY|Error"
  mkdir -p ${G}16; ls $G/gen_00*.npy | head -16 | xargs -I{} ln -sf {} ${G}16/
  trel gscheck --pred ${G}16 --out ${G}_render | grep -E "GSCHECK_DONE|Error"
  echo "  색: $($PY $PR/scripts/gs_summary.py ${G}_render/gscheck.jsonl)"
done
$PY - <<PYEOF
import glob, json, os
from PIL import Image, ImageDraw
O='$O'; S=170; rows=[('T1.0',O+'/gen_u'),('T0.8',O+'/gen_u_t0.8'),('T0.6',O+'/gen_u_t0.6')]
c=Image.new('RGB',(S*16,30+S*len(rows)),'white'); d=ImageDraw.Draw(c); d.text((4,4),'B prior samples by start-noise temperature (number = max cos to train tokens; real val renders ~0.876)',fill='black')
for r,(h,G) in enumerate(rows):
    cc=json.load(open(G+'/copy_check.json'))['gen']
    for i,f in enumerate(sorted(glob.glob(G+'_render/*_gs.png'))[:16]):
        im=Image.open(f).convert('RGB'); im=im.crop((0,0,im.height,im.height)).resize((S,S)); c.paste(im,(i*S,30+r*S))
        d.text((i*S+3,30+r*S+3),'%s %.2f'%(h if i==0 else '',cc[int(os.path.basename(f)[4:8])]),fill='blue')
c.save(O+'/gen_temp_비교.png'); print('MONTAGE')
PYEOF
echo B_TEMP_DONE
