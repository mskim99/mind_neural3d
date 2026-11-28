# V-10 (11-28): 진단 렌더의 질감 통계 12개 — v9_feat.py frame_feat의 질감 부분과 같은 식 (흰 배경 합성 → 512² → 마스크 안 L* · Hann · 대역 3 × 방향 4 log 에너지), 6뷰 평균
#  자극(10프레임 평균)과 같은 PCA에 넣어 생성 결과 PC1 · PC2를 만들 재료 (분석은 로컬)
# 출력: <out>/tex_<세트>.npz (uid, tex [n, 12])
# 사용: python -I v10_render_tex.py --diag <폴더> --sets a,b --out <폴더> [--workers 6]
import os, sys, argparse, numpy as np
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_v] = '1'
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser(); p.add_argument('--diag', required=True); p.add_argument('--sets', required=True); p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=6)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
R = 512
_fy, _fx = np.fft.fftfreq(R)[:, None] * R, np.fft.fftfreq(R)[None] * R
fr = np.hypot(_fy, _fx); ang = (np.degrees(np.arctan2(_fy, _fx)) + 180) % 180
BANDS = [(4, 16), (16, 48), (48, 160)]; HANN = np.outer(np.hanning(R), np.hanning(R))
ORI = [((ang >= o - 22.5) & (ang < o + 22.5)) | ((ang >= o + 157.5)) if o == 0 else ((ang >= o - 22.5) & (ang < o + 22.5)) for o in (0, 45, 90, 135)]
BMASK = [[(fr >= lo) & (fr < hi) & om for om in ORI] for lo, hi in BANDS]


def tex_of(f, cv2):   # f: BGR uint8 흰 배경 (v9_feat.frame_feat와 같은 질감 식)
    m0 = f.min(2) < 235
    if m0.sum() < 500: return None
    f = cv2.resize(f, (R, R), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m0.astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST).astype(bool)
    mi = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if mi.sum() < 50: mi = m
    L = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 0] * (100 / 255)
    Lf = np.where(mi, L, L[mi].mean()) - L[mi].mean(); P = np.abs(np.fft.fft2(Lf * HANN)) ** 2
    return np.array([float(np.log(P[bm].mean() + 1e-6)) for row in BMASK for bm in row])


def one(d):
    import cv2
    from v6_common import rgba_to_bgr_white
    out = []
    for v in range(6):
        fp = f'{d}/{v}.png'
        if not os.path.exists(fp): continue
        im = cv2.imdecode(np.fromfile(fp, np.uint8), cv2.IMREAD_UNCHANGED)
        if im is None: continue
        if im.ndim == 3 and im.shape[2] == 4: im = rgba_to_bgr_white(im[..., [2, 1, 0, 3]])
        t = tex_of(im, cv2)
        if t is not None: out.append(t)
    return np.mean(out, 0) if len(out) >= 4 else None


if __name__ == '__main__':
    for s in a.sets.split(','):
        o = f'{a.out}/tex_{s}.npz'; root = f'{a.diag}/{s}'
        if os.path.exists(o) or not os.path.isdir(root): continue
        us = sorted(os.listdir(root))
        with Pool(a.workers) as pool: Rr = pool.map(one, [f'{root}/{u}' for u in us])
        ok = [i for i, r in enumerate(Rr) if r is not None]
        np.savez(o, uid=np.array([us[i] for i in ok]), tex=np.stack([Rr[i] for i in ok]).astype(np.float32))
        print(f'TEX {s} {len(ok)} / {len(us)}', flush=True)
    print('TEX_DONE', flush=True)
