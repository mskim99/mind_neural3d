#!/bin/bash
# A안 2단계 정성 (11-10): 예측 이미지 임베딩(components.npz 'img') → unCLIP(감마 0.5) → DINO(rembg) → 외형(SLat), 형상은 그 모델 예측 토큰. 정성 14물체
# 사용: GPU=1 TAG=10k INF=$W/prism_img/infer_10k/sub-0001_keep bash run_img_quick.sh
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; UPY=$W/venv_unclip/bin/python; M=$W/prism_img; Q=$PR/scripts/qual14_uids.txt; O=$M/q_$TAG
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $O; export HF_HUB_OFFLINE=1
NORM=$($PY -c "import numpy as np,glob; print(np.mean([np.linalg.norm(np.load(f)) for f in glob.glob('$TGT/clipimg/*.npy')]))")
$PY -c "import numpy as np; z=np.load('$INF/components.npz'); np.savez('$O/pred_img.npz', **{str(k).split('/')[1]: v for k, v in zip(z['name'], z['img'])})"
$PY $PR/text_prompts.py --comp $INF/components.npz --out $O/prompt.json
$UPY $PR/unclip_tools.py gen --frames $TGT/frames --names $Q --out $O/img --embeds $O/pred_img.npz --norm $NORM --gamma 0.5 | grep -E "DONE|Error"
trel dino --frames $O/img --out $O/tok --rembg 1 | grep -E "DONE|Error"
for v in unclip text pred; do
  case $v in unclip) X="--slat_pred $O/tok";; text) X="--slat_text $O/prompt.json";; pred) X="";; esac
  [ $(cat $O/quick/$v/gscheck.jsonl 2>/dev/null | wc -l) -ge 14 ] || trel gscheck --pred $INF/tokens --out $O/quick/$v --names $Q $X | grep -E "GSCHECK_DONE|Error|없음"
  echo "  $v 색: $($PY $PR/scripts/gs_summary.py $O/quick/$v/gscheck.jsonl)"
done
$PY - <<PYEOF
import os, json
from PIL import Image, ImageDraw
Q=[l.strip() for l in open('$Q') if l.strip()]; S=180; O='$O'; P=json.load(open(O+'/prompt.json'))
cols=[('stimulus f0','$TGT/frames/{}/0.png',0),('unCLIP(pred emb)',O+'/img/{}/0.png',0),('pred tok',O+'/quick/pred/{}_gs.png',1),('text',O+'/quick/text/{}_gs.png',1),('unCLIP->SLat',O+'/quick/unclip/{}_gs.png',1),('upper: GT emb g0.5','$W/prism_unclip/diag/quick/uc_g05/{}_gs.png',1)]
c=Image.new('RGB',(S*len(cols),20+S*len(Q)),'white'); d=ImageDraw.Draw(c)
for j,(h,_,_) in enumerate(cols): d.text((j*S+4,4),h,fill='black')
for r,u in enumerate(Q):
    for j,(h,f,cr) in enumerate(cols):
        f=f.format(u)
        if not os.path.exists(f): continue
        im=Image.open(f).convert('RGB'); im=im.crop((0,0,im.height,im.height)) if cr else im
        c.paste(im.resize((S,S)),(j*S,20+r*S))
    d.text((3*S+4,20+r*S+S-14),P.get(u,''),fill='blue')
c.save(O+'/비교.png'); print('MONTAGE', O+'/비교.png')
PYEOF
echo IMG_QUICK_DONE
