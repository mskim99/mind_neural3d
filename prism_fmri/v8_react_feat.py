# V-8 반응 표 재료 (11-28, CPU, 학습 없음): 진단 렌더 폴더 <diag>/<세트>/<uid>/{0..5}.png 마다
#  feat  = V-5와 같은 저수준 특징 10개 (v6_common.render_feat, 6뷰 평균, 흰 배경 합성)
#  color = 색 코드 3개 (color_agree.py와 같은 정의: 정규채도 · rg_n · by_n, 알파 > 200 · min RGB < 235 · 5×5 깎기, 6뷰 합침)
#  bright= 전경(알파 > 200) RGB 평균 (0 ~ 255), chroma = (max − min) 평균
# 출력: <out>/react_<세트>.npz (uid, feat, color, bright, chroma)
# 사용: python -I v8_react_feat.py --diag <폴더> --sets a,b,… --out <폴더> [--workers 6]
import os, sys, argparse, numpy as np
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_v] = '1'
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser(); p.add_argument('--diag', required=True); p.add_argument('--sets', required=True); p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=6)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)


def one(d):
    import cv2
    from v6_common import render_feat
    from PIL import Image, ImageFilter
    f = render_feat(d, cv2); pxs, br, ch = [], [], []
    for v in range(6):
        fp = f'{d}/{v}.png'
        if not os.path.exists(fp): continue
        im = np.asarray(Image.open(fp).convert('RGBA')); rgb = im[..., :3].astype(np.float32); al = im[..., 3] > 200
        if al.sum() < 50: continue
        br.append(rgb[al].mean()); ch.append((rgb[al].max(1) - rgb[al].min(1)).mean())
        m = al & (rgb.min(2) < 235); m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
        if m.sum() > 50: pxs.append(rgb[m])
    if f is None or not pxs: return None
    px = np.concatenate(pxs); mx, mn = px.max(1), px.min(1); R, G, B = px.mean(0); S = R + G + B + 1e-6
    col = [((mx - mn) / np.maximum(mx, 1)).mean(), 3 * (R - G) / S, 3 * ((R + G) / 2 - B) / S]
    return np.array(f, np.float32), np.array(col, np.float32), float(np.mean(br)), float(np.mean(ch))


for s in a.sets.split(','):
    o = f'{a.out}/react_{s}.npz'; root = f'{a.diag}/{s}'
    if os.path.exists(o) or not os.path.isdir(root): continue
    us = sorted(os.listdir(root))
    with Pool(a.workers) as pool: R = pool.map(one, [f'{root}/{u}' for u in us])
    ok = [i for i, r in enumerate(R) if r is not None]
    np.savez(o, uid=np.array([us[i] for i in ok]), feat=np.stack([R[i][0] for i in ok]), color=np.stack([R[i][1] for i in ok]),
             bright=np.array([R[i][2] for i in ok]), chroma=np.array([R[i][3] for i in ok]))
    print(f'REACT {s} {len(ok)} / {len(us)}', flush=True)
print('REACT_DONE', flush=True)
