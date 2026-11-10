#!/bin/bash
# vshape_g0 피험자 7명 정성 (11-10, 체리피킹용, Pro6000 GPU 1): 피험자마다 test 104 추론 → GLB(TRELLIS 이미지판) → 렌더
#  렌더 = 공정평가 번들 render.py 그대로, R2 카메라(거리 1.5 · FOV 56 · 고도 30) 중 뷰 0 · 2 · 4(방위 0 · 120 · 240)만, 512², Metallic 0
#  GLB는 피험자 차례로, 렌더는 GLB가 끝난 피험자부터 뒤에서 (한 번에 하나)
#  묶음: 256² JPG(흰 배경)로 줄여 vshape_allsub.tar — <sub>/<uid>/{0,2,4}.jpg + GT/<uid>/{0,2,4}.jpg(번들 R2) (VSHAPE_ALLSUB_DONE)
# 사용: nohup bash run_vshape_allsub.sh > $W/logs/prism_vshape_allsub.log 2>&1 < /dev/null & disown
SUBS="sub-0001 sub-0002 sub-0003 sub-0005 sub-0006 sub-0007 sub-0008"
source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; CK=$W/prism_vshape/vshape_g0/model_final.pt; I=$W/prism_vshape/infer/vshape_g0; O=$W/prism_vshape/allsub
B=$W/tools/blender-4.2.3-linux-x64/blender; RP=$W/fair_eval/scripts/render.py; GT=$W/fair_eval/gt_render/R2
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
[ -f $CK ] && [ -f $RP ] || { echo "없음: $CK 또는 $RP (공정평가 번들 먼저)"; exit 1; }
cd $W/run; mkdir -p $O/render

render() {   # 피험자
  local s=$1 g=$I/$1_keep/glb
  [ $(find $O/render/$s -name '*.png' 2>/dev/null | wc -l) -ge 312 ] && { echo "렌더 이미 있음 $s"; return; }
  ls $g/*/*.glb > $O/list_$s.txt
  $B -b -t 4 -P $RP -- --list $O/list_$s.txt --out_root $O/render/$s --res 512 --light 8000 --camera_dist 1.5 --fov 56 \
     --views 0:30,120:30,240:30 --metallic0 > $W/logs/vshape_allsub_render_$s.log 2>&1
  echo "렌더 $s $(find $O/render/$s -name '*.png' | wc -l)장 (예상 312) $(date +%H:%M:%S)"
}

step "1 추론 · GLB → 렌더 (피험자 차례로)"
RJ=""
for s in $SUBS; do
  [ -f $I/${s}_keep/components.npz ] || $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $I --subs $s --perceptive keep | grep -E "범주|DONE|Error"
  [ $(ls $I/${s}_keep/glb/*/*.glb 2>/dev/null | wc -l) -ge 104 ] || trel gen --pred $I/${s}_keep/tokens --out $I/${s}_keep/glb | grep -E "GEN_DONE|Error"
  echo "GLB $s $(ls $I/${s}_keep/glb/*/*.glb 2>/dev/null | wc -l)개 $(date +%H:%M:%S)"
  [ -n "$RJ" ] && wait $RJ
  render $s & RJ=$!
done
wait $RJ

step "2 묶기 (256² JPG)"
$PY - "$O" "$GT" $SUBS <<'EOF'
import sys, os, glob
from PIL import Image
O, GT, subs = sys.argv[1], sys.argv[2], sys.argv[3:]
P = f'{O}/pack'; n = 0
def save(src, dst):
    global n
    a = Image.open(src).convert('RGBA'); b = Image.new('RGBA', a.size, (255, 255, 255, 255)); b.paste(a, (0, 0), a)
    os.makedirs(os.path.dirname(dst), exist_ok=True); b.convert('RGB').resize((256, 256)).save(dst, quality=90); n += 1
for s in subs:
    for d in sorted(glob.glob(f'{O}/render/{s}/*/')):
        u = os.path.basename(d.rstrip('/'))
        for v in (0, 1, 2):
            if os.path.exists(f'{d}{v}.png'): save(f'{d}{v}.png', f'{P}/{s}/{u}/{2 * v}.jpg')
for d in sorted(glob.glob(f'{GT}/*/')):
    u = os.path.basename(d.rstrip('/'))
    for v in (0, 2, 4):
        if os.path.exists(f'{d}{v}.png'): save(f'{d}{v}.png', f'{P}/GT/{u}/{v}.jpg')
print('JPG', n, '(예상', 104 * 3 * (len(subs) + 1), ')')
EOF
for s in $SUBS; do cp $I/${s}_keep/components.npz $O/pack/$s/components.npz 2>/dev/null; done
cd $O && tar cf vshape_allsub.tar pack && echo "PACK $(du -sh vshape_allsub.tar | cut -f1)"
step "VSHAPE_ALLSUB_DONE"
