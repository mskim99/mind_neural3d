#!/bin/bash
# B안 2단계 (11-10, 5090): 압축 공간 prior 몇 개 → 생성 64 · 복사 검사 → 생성 16개 렌더 → 몽타주
#  GPU 0: glob (전역 PCA 코드) 여러 설정 / GPU 1: grid (채널 PCA 64 × 19²) DiT — 복원 렌더(run_b_comp.sh) 끝난 뒤
# 사용: nohup bash run_b_comp_prior.sh > $W/logs/prism_b_comp_prior.log 2>&1 < /dev/null & disown
GPU=0 source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_b; C=$O/comp
PC="$PY $PR/prior_comp.py --comp $C --tok $O/tok --bankdir $O/prior_u"
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run
run() {   # run <gpu> <이름> <공간 인자> <학습 인자>
  g=$1; n=$2; sp="$3"; tr="$4"; D=$C/$n
  echo "=== $n 학습 $(date +%H:%M:%S) [$sp $tr]"
  [ -f $D/prior_final.pt ] || CUDA_VISIBLE_DEVICES=$g $PC train $sp $tr --out $D > $D.log 2>&1
  grep -E "\[model\]|\[val\]|Error" $D.log | tail -4
  CUDA_VISIBLE_DEVICES=$g $PC sample $sp --ckpt $D/prior_final.pt --out $D/gen --n 64 | grep -E "COPY|Error"
  mkdir -p $D/gen16; ls $D/gen/gen_00*.npy | head -16 | xargs -I{} ln -sf {} $D/gen16/
  CUDA_VISIBLE_DEVICES=$g trel gscheck --pred $D/gen16 --out $D/render | grep -E "GSCHECK_DONE|Error"
  echo "=== $n 끝 $(date +%H:%M:%S)"
}
gpu0() {
  while pgrep -f "[p]g512 " >/dev/null || pgrep -f "out prism_b/comp/[p]g512" >/dev/null; do sleep 20; done
  run 0 pg512 "--space glob --cfg c64_s2 --k 512" "--width 1024 --depth 6 --bs 256 --steps 30000"
  run 0 pg256n "--space glob --cfg c64_s2 --k 256 --white 0" "--width 512 --depth 4 --dropout 0.3 --bs 256 --steps 10000"
  run 0 pg1024n "--space glob --cfg c64_s2 --k 1024 --white 0" "--width 512 --depth 4 --dropout 0.3 --bs 256 --steps 10000"
}
gpu1() {
  while pgrep -f "[r]un_b_comp.sh" >/dev/null; do sleep 20; done
  run 1 pgrid "--space grid --cfg c64_s2" "--width 384 --depth 8 --heads 6 --dropout 0 --bs 64 --steps 15000"
}
gpu0 & gpu1 & wait
$PY - <<PYEOF
import json, glob, os
from PIL import Image, ImageDraw
C='$C'; S=170; runs=['pg512','pg256n','pg1024n','pgrid']
c=Image.new('RGB',(S*16,40+S*len(runs)),'white'); d=ImageDraw.Draw(c)
d.text((4,4),'B compressed-space prior samples (unconditional) - TRELLIS render; number = max cos to train tokens',fill='black')
for r,n in enumerate(runs):
    try: cc=json.load(open(f'{C}/{n}/gen/copy_check.json'))
    except Exception: continue
    d.text((4,24+r*S),f"{n}: mean {cc['mean_gen']:.3f} (recon ceiling {cc['mean_recon_val']:.3f}, real {cc['mean_real_val']:.3f}) pair {cc['gen_pair_cos']:.3f}",fill='black')
    for i,f in enumerate(sorted(glob.glob(f'{C}/{n}/render/*_gs.png'))[:16]):
        im=Image.open(f).convert('RGB'); im=im.crop((0,0,im.height,im.height)).resize((S,S-14)); c.paste(im,(i*S,40+r*S))
        d.text((i*S+4,40+r*S+2),'%.2f'%cc['gen'][int(os.path.basename(f)[4:8])],fill='blue')
c.save(C+'/압축prior_생성.png'); print('MONTAGE', C+'/압축prior_생성.png')
PYEOF
echo "=== B_COMP_PRIOR_DONE $(date +%H:%M:%S)"
