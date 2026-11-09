# V "변하는 정보" 정답 준비 (11-09, 로컬 CPU): fMRI 프레임 2~7에 대응하는 자극 영상 프레임(round(19.2k) = 38 · 58 · 77 · 96 · 115 · 134)마다
#  ① 실루엣 값 [cx, cy, w, h, area] — 흰 배경 영상에서 min(RGB) < 235 (prep_shape_targets.py와 같은 규칙). cx · cy ∈ [-1, 1] (bbox 중심), w · h = bbox 폭 · 높이 / 화면, area = 물체 픽셀 비율
#  ② DINO용 프레임: bbox 둘레 정사각 × 1.3으로 자르고(흰 배경으로 채움) 518²로 줄인 PNG → 서버 trellis_tools.py dino --pool 1 (시점 특징 정답)
#  "물체 평균 빼기"(변하는 부분만)는 학습 때 train 물체 통계로 함 (train_prism.py --var_heads sil|view). 여기서는 원래 값만 저장
# 출력: <out>/view_sil.npz (keys [N], vf [6], frames [6], sil [N, 6, 5]) · <out>/vframes/<uid>/{2..7}.png
# 사용: python -I prep_view_targets.py --data <DataSet/MinD-3D++> --out <폴더> [--workers 10] [--limit N]
import os, sys, zipfile, argparse, tempfile
import numpy as np
from multiprocessing import Pool
p = argparse.ArgumentParser()
p.add_argument('--data', default=r'C:\Users\jinsw712\Desktop\Files\EEG\DataSet\MinD-3D++')
p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=10); p.add_argument('--limit', type=int, default=0)
p.add_argument('--no_frames', type=int, default=0)
a = p.parse_args()
SH = os.path.join(a.data, 'fMRI-Shape')
KEYS = [l.strip() for f in ('core_train_list.txt', 'core_test_list.txt') for l in open(os.path.join(SH, 'annotations', f)) if l.strip()]
if a.limit: KEYS = KEYS[:a.limit]
FRAMES = list(range(2, 8)); VF = [min(191, round(19.2 * f)) for f in FRAMES]


def work(key):
    import cv2
    uid = key.split('/')[1]
    z = zipfile.ZipFile(os.path.join(SH, 'stimuli.zip'))
    fd, tmp = tempfile.mkstemp(suffix='.mp4'); os.close(fd)
    out = np.full((len(VF), 5), np.nan, np.float32)
    try:
        open(tmp, 'wb').write(z.read(f'stimuli/{key}.mp4')); c = cv2.VideoCapture(tmp); i = 0
        while i <= max(VF):
            ok, f = c.read()
            if not ok: break
            if i in VF:
                j = VF.index(i); H, W = f.shape[:2]; m = f.min(2) < 235
                ys, xs = np.nonzero(m)
                if len(ys) > 50:
                    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                    out[j] = [(x0 + x1) / W - 1, (y0 + y1) / H - 1, (x1 - x0 + 1) / W, (y1 - y0 + 1) / H, m.mean()]
                    if not a.no_frames:
                        s = int(max(x1 - x0, y1 - y0) * 1.3) + 2; cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                        can = np.full((s, s, 3), 255, np.uint8)
                        sx0, sy0 = cx - s // 2, cy - s // 2
                        a0, b0 = max(0, sx0), max(0, sy0); a1, b1 = min(W, sx0 + s), min(H, sy0 + s)
                        can[b0 - sy0:b1 - sy0, a0 - sx0:a1 - sx0] = f[b0:b1, a0:a1]
                        d = os.path.join(a.out, 'vframes', uid); os.makedirs(d, exist_ok=True)
                        cv2.imwrite(os.path.join(d, f'{FRAMES[j]}.png'), cv2.resize(can, (518, 518), interpolation=cv2.INTER_AREA))
            i += 1
        c.release()
    finally:
        os.remove(tmp)
    return out


if __name__ == '__main__':
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        res = []
        for n, r in enumerate(pool.imap(work, KEYS, chunksize=4)):
            res.append(r)
            if (n + 1) % 200 == 0: print(f'[view] {n + 1}/{len(KEYS)}', flush=True)
    S = np.stack(res)
    np.savez(os.path.join(a.out, 'view_sil.npz'), keys=np.array(KEYS), vf=np.array(VF), frames=np.array(FRAMES), sil=S)
    bad = int(np.isnan(S).any(-1).sum())
    print(f'VIEW_DONE {len(KEYS)} 물체 · 빈 프레임 {bad} · sil 평균 {np.nanmean(S, (0, 1)).round(3).tolist()} · 프레임 간 표준편차(물체 안 평균) {np.nanmean(np.nanstd(S, 1), 0).round(4).tolist()}', flush=True)
