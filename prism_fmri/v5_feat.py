# V-5 (11-24): 자극 영상의 저수준 외형 특징 — 물체마다 하나 (로컬 CPU, 학습 없음)
#  프레임: 영상 0 · 19 · 38 · 58 · 77 · 96 · 115 · 134 · 154 · 173 = round(19.2 × k) (fMRI 10프레임 시점, silhouettes32.npz video_frames와 같음) → 특징을 프레임 평균
#  마스크: min(R, G, B) < 235 (prep_shape_targets.py와 같은 규칙). 외형 특징은 테두리(흰 배경과의 경계) 영향을 빼려고 마스크를 6 px(1024² 기준) 깎은 안쪽에서 잼
#  특징 (512²로 줄여서 계산):
#   L_mean 평균 밝기(CIELAB L*) · L_std 대비(L* 표준편차) · grad 질감 세기(L* Sobel 기울기 크기 평균)
#   band_lo · band_mid · band_hi 공간 주파수 에너지(물체 안 L*, 밖은 물체 평균으로 채우고 창 씌운 FFT, 화면 기준 4 ~ 16 · 16 ~ 48 · 48 ~ 160 주기, log)
#   chroma 평균 채도(√(a*² + b*²)) · a_mean · b_mean 색상 · area 마스크 면적 비율(대조: 이미 읽히는 값)
# 출력: <out> npz — keys [N], feat [N, 10], names [10], frames [10]
# 사용: python -I v5_feat.py --out <npz> [--workers 8]
import os, io, zipfile, argparse, tempfile
import numpy as np
from multiprocessing import Pool
p = argparse.ArgumentParser()
p.add_argument('--data', default=r'C:\Users\jinsw712\Desktop\Files\EEG\DataSet\MinD-3D++')
p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=8); p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
SH = os.path.join(a.data, 'fMRI-Shape')
IDS = [l.strip() for f in ('core_train_list.txt', 'core_test_list.txt') for l in open(os.path.join(SH, 'annotations', f)) if l.strip()][: a.limit or None]
VF = [int(round(19.2 * k)) for k in range(10)]
NAMES = ['L_mean', 'L_std', 'grad', 'band_lo', 'band_mid', 'band_hi', 'chroma', 'a_mean', 'b_mean', 'area']
R = 512; yy, xx = np.mgrid[:R, :R]; fr = np.hypot(np.fft.fftfreq(R)[:, None], np.fft.fftfreq(R)[None]) * R   # 화면 기준 주기
BANDS = [(4, 16), (16, 48), (48, 160)]; HANN = np.outer(np.hanning(R), np.hanning(R))


def frame_feat(f, cv2):
    m0 = f.min(2) < 235
    if m0.sum() < 500: return None
    f = cv2.resize(f, (R, R), interpolation=cv2.INTER_AREA); area = float(m0.mean())
    m = cv2.resize(m0.astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST).astype(bool)
    mi = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)          # 512²에서 3 px = 1024²에서 6 px
    if mi.sum() < 50: mi = m
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0] * (100 / 255); A = lab[..., 1] - 128; B = lab[..., 2] - 128
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3); g = np.hypot(gx, gy)
    Lf = np.where(mi, L, L[mi].mean()) - L[mi].mean(); P = np.abs(np.fft.fft2(Lf * HANN)) ** 2
    bands = [float(np.log(P[(fr >= lo) & (fr < hi)].mean() + 1e-6)) for lo, hi in BANDS]
    return [float(L[mi].mean()), float(L[mi].std()), float(g[mi].mean())] + bands + [float(np.hypot(A[mi], B[mi]).mean()), float(A[mi].mean()), float(B[mi].mean()), area]


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
        return k, (np.mean(out, 0) if len(out) >= 5 else None), len(out)
    except Exception as e:
        return k, None, repr(e)[:100]


if __name__ == '__main__':
    with Pool(a.workers) as pool:
        res = {}
        for n, (k, v, info) in enumerate(pool.imap_unordered(work, IDS, chunksize=4)):
            res[k] = v
            if v is None: print('FAIL', k, info, flush=True)
            if n % 100 == 0: print(f'[{n + 1}/{len(IDS)}] {k} 프레임 {info}', flush=True)
    keys = [k for k in IDS if res.get(k) is not None]
    np.savez(a.out, keys=np.array(keys), feat=np.stack([res[k] for k in keys]).astype(np.float32), names=np.array(NAMES), frames=np.array(VF))
    print('V5_FEAT_DONE', len(keys), '/', len(IDS), a.out, flush=True)
