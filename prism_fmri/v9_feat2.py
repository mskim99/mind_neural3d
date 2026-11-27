# V-9 추가 (11-28, 총괄 04:31): 자리 · 크기를 뺀 흐릿한 이미지, 고주파 에너지의 공간 지도 — 로컬 CPU, 학습 없음 (v9_feat.py와 같은 프레임 · 마스크 규칙)
#  ② crop8_avg · cmask8_avg: TRELLIS preprocess_image와 같은 자르기 — 알파(min RGB < 235) bbox의 중심, 한 변 = max(폭, 높이) × 1.2 정사각형(밖은 흰색으로 채움)
#     → 8×8 RGB(INTER_AREA) · 같은 자르기의 마스크 8×8 → 10프레임 평균
#  ③ hbmap4_avg · hbmap8_avg: 물체 안 L*(v5_feat처럼 마스크 6 px 깎은 안쪽, 밖은 물체 평균으로 채움, 512²)를 48 ~ 160 주기 대역으로 거름(FFT, 창 없음)
#     → |값|² → 4×4 · 8×8 칸 평균 → log(+1e-3) → 10프레임 평균. 화면 전체 기준(자르지 않음). 마스크 몫을 뺄 때 쓰도록 mask4_avg(화면 전체 4×4)도 저장
# 출력: <out> npz — keys, crop8_avg [N,8,8,3], cmask8_avg [N,8,8], hbmap4_avg [N,4,4], hbmap8_avg [N,8,8], mask4_avg [N,4,4]
# 사용: python -I v9_feat2.py --out <npz> [--workers 10]
import os, zipfile, argparse, tempfile
import numpy as np
from multiprocessing import Pool
p = argparse.ArgumentParser()
p.add_argument('--data', default=r'C:\Users\jinsw712\Desktop\Files\EEG\DataSet\MinD-3D++')
p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=10); p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
SH = os.path.join(a.data, 'fMRI-Shape')
IDS = [l.strip() for f in ('core_train_list.txt', 'core_test_list.txt') for l in open(os.path.join(SH, 'annotations', f)) if l.strip()][: a.limit or None]
VF = [int(round(19.2 * k)) for k in range(10)]
R = 512; fr = np.hypot(np.fft.fftfreq(R)[:, None], np.fft.fftfreq(R)[None]) * R; HB = (fr >= 48) & (fr < 160)


def frame_feat(f, cv2):
    m0 = f.min(2) < 235
    if m0.sum() < 500: return None
    ys, xs = np.where(m0); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2; s = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.2) // 2
    H, W = m0.shape; pad = s + 2
    fp = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)); mp = np.pad(m0, pad)
    y0, x0 = int(cy) - s + pad, int(cx) - s + pad
    cr = fp[y0:y0 + 2 * s, x0:x0 + 2 * s][..., ::-1].astype(np.float32) / 255; cm = mp[y0:y0 + 2 * s, x0:x0 + 2 * s].astype(np.float32)
    c8 = cv2.resize(cr, (8, 8), interpolation=cv2.INTER_AREA); cm8 = cv2.resize(cm, (8, 8), interpolation=cv2.INTER_AREA)
    m4 = cv2.resize(m0.astype(np.float32), (4, 4), interpolation=cv2.INTER_AREA)
    g = cv2.resize(f, (R, R), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m0.astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST).astype(bool)
    mi = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if mi.sum() < 50: mi = m
    L = cv2.cvtColor(g, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 0] * (100 / 255)
    Lf = np.where(mi, L, L[mi].mean()) - L[mi].mean(); E = np.abs(np.fft.ifft2(np.fft.fft2(Lf) * HB)) ** 2
    h4 = np.log(cv2.resize(E.astype(np.float32), (4, 4), interpolation=cv2.INTER_AREA) + 1e-3); h8 = np.log(cv2.resize(E.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA) + 1e-3)
    return c8, cm8, h4, h8, m4


def work(k):
    import cv2
    try:
        z = zipfile.ZipFile(os.path.join(SH, 'stimuli.zip'))
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as t: t.write(z.read(f'stimuli/{k}.mp4')); tmp = t.name
        cap = cv2.VideoCapture(tmp); out, i = [], 0
        while i <= VF[-1]:
            ok, f = cap.read()
            if not ok: break
            if i in VF:
                r = frame_feat(f, cv2)
                if r is not None: out.append(r)
            i += 1
        cap.release(); os.remove(tmp)
        if len(out) < 5: return k, None, len(out)
        return k, [np.mean([o[j] for o in out], 0) for j in range(5)], len(out)
    except Exception as e:
        return k, None, repr(e)[:100]


if __name__ == '__main__':
    with Pool(a.workers) as pool:
        res = {}
        for n, (k, v, info) in enumerate(pool.imap_unordered(work, IDS, chunksize=4)):
            res[k] = v
            if v is None: print('FAIL', k, info, flush=True)
            if n % 200 == 0: print(f'[{n + 1}/{len(IDS)}] {k} {info}', flush=True)
    keys = [k for k in IDS if res.get(k) is not None]
    st = lambda j: np.stack([res[k][j] for k in keys]).astype(np.float32)
    np.savez(a.out, keys=np.array(keys), crop8_avg=st(0), cmask8_avg=st(1), hbmap4_avg=st(2), hbmap8_avg=st(3), mask4_avg=st(4), frames=np.array(VF))
    print('V9_FEAT2_DONE', len(keys), '/', len(IDS), a.out, flush=True)
