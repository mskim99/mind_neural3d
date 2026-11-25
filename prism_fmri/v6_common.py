# V-6 (11-25) 공통: V-5와 같은 저수준 외형 특징을 렌더 이미지에 잰다 (v5_feat.py frame_feat와 같은 식 · 같은 상수)
#  입력: BGR uint8 이미지(흰 배경). RGBA 렌더는 rgba_to_bgr_white로 흰 배경에 합성해서 넣는다
#  출력: [L_mean, L_std, grad, band_lo, band_mid, band_hi, chroma, a_mean, b_mean, area] 또는 None(물체 너무 작음)
import numpy as np
NAMES = ['L_mean', 'L_std', 'grad', 'band_lo', 'band_mid', 'band_hi', 'chroma', 'a_mean', 'b_mean', 'area']
R = 512
fr = np.hypot(np.fft.fftfreq(R)[:, None], np.fft.fftfreq(R)[None]) * R
BANDS = [(4, 16), (16, 48), (48, 160)]; HANN = np.outer(np.hanning(R), np.hanning(R))


def rgba_to_bgr_white(rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) * a + 255 * (1 - a)
    return np.ascontiguousarray(rgb[..., ::-1].round().clip(0, 255).astype(np.uint8))


def frame_feat(f, cv2):
    m0 = f.min(2) < 235
    if m0.sum() < 500: return None
    f = cv2.resize(f, (R, R), interpolation=cv2.INTER_AREA); area = float(m0.mean())
    m = cv2.resize(m0.astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST).astype(bool)
    mi = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if mi.sum() < 50: mi = m
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0] * (100 / 255); A = lab[..., 1] - 128; B = lab[..., 2] - 128
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3); g = np.hypot(gx, gy)
    Lf = np.where(mi, L, L[mi].mean()) - L[mi].mean(); P = np.abs(np.fft.fft2(Lf * HANN)) ** 2
    bands = [float(np.log(P[(fr >= lo) & (fr < hi)].mean() + 1e-6)) for lo, hi in BANDS]
    return [float(L[mi].mean()), float(L[mi].std()), float(g[mi].mean())] + bands + [float(np.hypot(A[mi], B[mi]).mean()), float(A[mi].mean()), float(B[mi].mean()), area]


def render_feat(folder, cv2, nview=6):
    """<folder>/{0..nview-1}.png (RGBA) → 뷰 평균 특징 (V-5의 프레임 평균과 같은 방식). 4뷰 미만이면 None"""
    import os
    out = []
    for v in range(nview):
        p = os.path.join(folder, f'{v}.png')
        if not os.path.exists(p): continue
        im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)   # 한글 경로(Windows)도 읽히게
        if im is None: continue
        if im.ndim == 3 and im.shape[2] == 4: im = rgba_to_bgr_white(im[..., [2, 1, 0, 3]])   # cv2는 BGRA → RGBA로 바꿔 합성
        r = frame_feat(im, cv2)
        if r is not None: out.append(r)
    return np.mean(out, 0) if len(out) >= 4 else None
