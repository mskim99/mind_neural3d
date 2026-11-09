# B안 렌더 보정 (11-09): render_b.py calib 실루엣 vs 자극 프레임 0 실루엣 IoU → yaw · fov 고르기
#  자극 실루엣 = 흰 배경 min(RGB) < 235 (prep_shape_targets.py와 같음), 렌더 실루엣 = 알파 > 127, 둘 다 256²
#  fMRI-Shape 목록 uid는 잘린 것이 있어 frames 폴더 이름이 ShapeNet uid의 앞부분인지로 대조
# 사용: python calib_b.py pick --objects <objects_b.txt> --train <core_train_list.txt> --per 2 --out <calib 목록>
#       python calib_b.py score --calib <render 폴더> --frames <tgt>/frames
import os, sys, glob, argparse, numpy as np
from PIL import Image
p = argparse.ArgumentParser(); p.add_argument('mode', choices=['pick', 'score'])
p.add_argument('--objects'); p.add_argument('--train'); p.add_argument('--per', type=int, default=2); p.add_argument('--out')
p.add_argument('--calib'); p.add_argument('--frames'); p.add_argument('--res', type=int, default=256)
a = p.parse_args()
if a.mode == 'pick':
    tr = [l.strip() for l in open(a.train) if l.strip()]; obj = [l.strip() for l in open(a.objects) if l.strip()]
    by = {}
    for o in obj:
        if any(o.startswith(t) for t in tr): by.setdefault(o.split('/')[0], []).append(o)
    sel = [x for c in sorted(by) for x in sorted(by[c])[: a.per]]
    open(a.out, 'w').write('\n'.join(sel) + '\n'); print('calib', len(sel), '→', a.out)
    sys.exit()
fr = sorted(os.listdir(a.frames)); S = a.res
def stim(uid):
    full = uid.split('_')[1]; m = [f for f in fr if full.startswith(f)]
    if not m: return None
    x = np.asarray(Image.open(f'{a.frames}/{m[0]}/0.png').convert('RGB').resize((S, S), Image.BILINEAR))
    return x.min(2) < 235
tab = {}
for od in sorted(glob.glob(f'{a.calib}/*')):
    g = stim(os.path.basename(od))
    if g is None: continue
    for f in glob.glob(f'{od}/y*_f*.png'):
        r = np.asarray(Image.open(f).convert('RGBA').resize((S, S), Image.BILINEAR))[..., 3] > 127
        iou = (g & r).sum() / max((g | r).sum(), 1)
        tab.setdefault(os.path.basename(f)[:-4], []).append(iou)
rows = sorted(((np.mean(v), k, len(v)) for k, v in tab.items()), reverse=True)
for m, k, nn in rows[:12]: print(f'{k:>12}  IoU {m:.3f}  (n {nn})')
yaws = sorted({k.split('_')[0] for k in tab})
for y in yaws:
    b = max((np.mean(v), k) for k, v in tab.items() if k.startswith(y + '_')); print(f'  {y} 최고 {b[1]} {b[0]:.3f}')
print('CALIB_BEST', rows[0][1], f'{rows[0][0]:.3f}')
