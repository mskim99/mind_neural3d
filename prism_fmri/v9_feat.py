# V-9 (11-28): Variational 구조 후보의 정답 — 자극 영상에서 물체마다 (로컬 CPU, 학습 없음). v5_feat.py와 같은 프레임 · 마스크 규칙
#  프레임: fMRI 10프레임 시점 0 · 19 · … · 173 (v5_feat.py VF)
#  ① 흐릿한 이미지: 프레임 전체(1024², 흰 배경)를 INTER_AREA로 8×8 · 16×16 RGB(0 ~ 1)로 줄임 — 프레임 0 / 10프레임 평균
#  ③ 색 분포: 물체 안쪽(마스크 6 px 깎음) 평균 Lab(L* 0 ~ 100, a* · b*) + a* · b* 4×4 histogram(각 −48 ~ 48, 24 간격, 밖은 끝 칸, 비율) → 10프레임 평균
#  ④ 질감 통계: v5_feat.py의 대역 3개(4 ~ 16 · 16 ~ 48 · 48 ~ 160 주기)를 방향 4개(0 · 45 · 90 · 135°, ±22.5°)로 나눈 log 에너지 12개 → 10프레임 평균
#     (방향 = 주파수 벡터의 각도 mod 180, 0° = 가로 주파수 = 세로 줄무늬)
# 출력: <out> npz — keys [N], blur8_f0 [N,8,8,3], blur16_f0 [N,16,16,3], blur8_avg, blur16_avg, lab_mean [N,3], ab_hist [N,16], tex_ob [N,12] (대역 × 방향), frames
# 사용: python -I v9_feat.py --out <npz> [--workers 10]
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
R = 512
_fy, _fx = np.fft.fftfreq(R)[:, None] * R, np.fft.fftfreq(R)[None] * R
fr = np.hypot(_fy, _fx); ang = (np.degrees(np.arctan2(_fy, _fx)) + 180) % 180
BANDS = [(4, 16), (16, 48), (48, 160)]; HANN = np.outer(np.hanning(R), np.hanning(R))
ORI = [((ang >= o - 22.5) & (ang < o + 22.5)) | ((ang >= o + 157.5)) if o == 0 else ((ang >= o - 22.5) & (ang < o + 22.5)) for o in (0, 45, 90, 135)]
BMASK = [[(fr >= lo) & (fr < hi) & om for om in ORI] for lo, hi in BANDS]
EDGES = np.array([-48, -24, 0, 24, 48])


def frame_feat(f, cv2):
    rgb = f[..., ::-1].astype(np.float32) / 255
    b8 = cv2.resize(rgb, (8, 8), interpolation=cv2.INTER_AREA); b16 = cv2.resize(rgb, (16, 16), interpolation=cv2.INTER_AREA)
    m0 = f.min(2) < 235
    if m0.sum() < 500: return b8, b16, None
    f = cv2.resize(f, (R, R), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m0.astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST).astype(bool)
    mi = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if mi.sum() < 50: mi = m
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0] * (100 / 255); A = lab[..., 1] - 128; B = lab[..., 2] - 128
    ia = np.clip(np.digitize(A[mi], EDGES[1:-1]), 0, 3); ib = np.clip(np.digitize(B[mi], EDGES[1:-1]), 0, 3)
    h = np.bincount(ia * 4 + ib, minlength=16).astype(np.float64); h /= h.sum()
    Lf = np.where(mi, L, L[mi].mean()) - L[mi].mean(); P = np.abs(np.fft.fft2(Lf * HANN)) ** 2
    tex = [float(np.log(P[bm].mean() + 1e-6)) for row in BMASK for bm in row]
    return b8, b16, np.concatenate([[L[mi].mean(), A[mi].mean(), B[mi].mean()], h, tex])


def work(k):
    import cv2
    try:
        z = zipfile.ZipFile(os.path.join(SH, 'stimuli.zip'))
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as t: t.write(z.read(f'stimuli/{k}.mp4')); tmp = t.name
        cap = cv2.VideoCapture(tmp); b8s, b16s, vs, i = [], [], [], 0
        while i <= VF[-1]:
            ok, f = cap.read()
            if not ok: break
            if i in VF:
                b8, b16, v = frame_feat(f, cv2); b8s.append(b8); b16s.append(b16)
                if v is not None: vs.append(v)
            i += 1
        cap.release(); os.remove(tmp)
        if len(vs) < 5 or len(b8s) < 10: return k, None, f'프레임 {len(b8s)} · 물체 {len(vs)}'
        return k, (b8s[0], b16s[0], np.mean(b8s, 0), np.mean(b16s, 0), np.mean(vs, 0)), len(vs)
    except Exception as e:
        return k, None, repr(e)[:100]


if __name__ == '__main__':
    with Pool(a.workers) as pool:
        res = {}
        for n, (k, v, info) in enumerate(pool.imap_unordered(work, IDS, chunksize=4)):
            res[k] = v
            if v is None: print('FAIL', k, info, flush=True)
            if n % 100 == 0: print(f'[{n + 1}/{len(IDS)}] {k} {info}', flush=True)
    keys = [k for k in IDS if res.get(k) is not None]
    st = lambda j: np.stack([res[k][j] for k in keys]).astype(np.float32)
    V = st(4)
    np.savez(a.out, keys=np.array(keys), blur8_f0=st(0), blur16_f0=st(1), blur8_avg=st(2), blur16_avg=st(3),
             lab_mean=V[:, :3], ab_hist=V[:, 3:19], tex_ob=V[:, 19:31], frames=np.array(VF))
    print('V9_FEAT_DONE', len(keys), '/', len(IDS), a.out, flush=True)
