# B안 렌더 색 보정 (11-09): 렌더(설정별 폴더) vs 자극 프레임 0 — 전경(흰 배경 min(RGB) < 235) 평균 밝기 · 채도와 물체별 평균 RGB 차이
# 사용: python calib_look_b.py --frames <tgt>/frames --dirs <폴더1>,<폴더2>,...
import os, argparse, numpy as np
from PIL import Image
p = argparse.ArgumentParser(); p.add_argument('--frames', required=True); p.add_argument('--dirs', required=True); a = p.parse_args()
fr = sorted(os.listdir(a.frames))
def stat(f):
    x = np.asarray(Image.open(f).convert('RGB').resize((256, 256))).astype(np.float32); m = x.min(2) < 235
    px = x[m] if m.sum() > 20 else x.reshape(-1, 3)
    return px.mean(0), px.mean(), (px.max(1) - px.min(1)).mean()
for d in a.dirs.split(','):
    rows = []
    for u in sorted(os.listdir(d)):
        g = [f for f in fr if u.split('_')[1].startswith(f)]
        if not g or not os.path.exists(f'{d}/{u}/0.png'): continue
        sr, sb, sc = stat(f'{a.frames}/{g[0]}/0.png'); rr, rb, rc = stat(f'{d}/{u}/0.png')
        rows.append((rb, sb, rc, sc, np.abs(rr - sr).mean()))
    r = np.array(rows)
    print(f'{os.path.basename(d):>16} n {len(r)} | 밝기 렌더 {r[:,0].mean():.1f} vs 자극 {r[:,1].mean():.1f} | 채도 {r[:,2].mean():.1f} vs {r[:,3].mean():.1f} | 물체별 RGB 차 {r[:,4].mean():.1f}', flush=True)
