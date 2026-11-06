# 렌더 색 통계 (11-06 색 안 3): 평가 렌더 폴더마다 물체 영역의 평균 밝기 · 채도(max−min), 물체별 평균 → 전체 평균
#  물체 영역 = RGBA면 alpha > 128, 아니면 min(RGB) < 245. 참고값 = 자극 영상 프레임 0 (<tgt>/frames/<uid>/0.png, min RGB < 235)
# 사용: python color_stats.py --tgt <shape_targets> --ids <core_test_list.txt> 이름=렌더폴더 [이름=렌더폴더 ...]
import os, sys, argparse, numpy as np
from PIL import Image
p = argparse.ArgumentParser(); p.add_argument('--tgt', required=True); p.add_argument('--ids', required=True); p.add_argument('runs', nargs='+')
a = p.parse_args()


def stats(px):
    return float(px.mean()), float((px.max(1) - px.min(1)).mean())


def img_px(f):
    im = np.asarray(Image.open(f)); rgb = im[..., :3].astype(np.float32)
    m = im[..., 3] > 128 if im.shape[-1] == 4 else rgb.min(2) < 245
    return rgb[m] if m.sum() > 50 else None


uids = [l.strip().split('/')[1] for l in open(a.ids) if l.strip()]
fr = lambda u: np.asarray(Image.open(f'{a.tgt}/frames/{u}/0.png').convert('RGB')).reshape(-1, 3).astype(np.float32)
ref = [stats(r[r.min(1) < 235]) for r in map(fr, uids)]
print(f'COLOR 자극 프레임0      밝기 {np.mean([r[0] for r in ref]):6.1f} 채도 {np.mean([r[1] for r in ref]):5.1f} (n {len(ref)})', flush=True)
for run in a.runs:
    name, root = run.split('=', 1); dirs = {}
    for d, _, fs in os.walk(root):
        if '0.png' in fs:
            u = next((x for x in uids if os.path.basename(d).startswith(x) or x in d), None)
            if u: dirs.setdefault(u, d)
    vals = []
    for u, d in dirs.items():
        px = [x for x in (img_px(f'{d}/{f}') for f in sorted(os.listdir(d)) if f.endswith('.png')) if x is not None]
        if px: vals.append(stats(np.concatenate(px)))
    print(f'COLOR {name:<16} 밝기 {np.mean([v[0] for v in vals]):6.1f} 채도 {np.mean([v[1] for v in vals]):5.1f} (n {len(vals)})' if vals else f'COLOR {name} 렌더 없음 {root}', flush=True)
