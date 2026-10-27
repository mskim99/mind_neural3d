# PRISM v2 (박사님 10-27 구조도) 전처리 — fMRI-Shape core 1,404개 (train 1,300 + test 104), 로컬 CPU
#  ① Variational 정답 (자극 영상에서, 사용자 10-27): 영상 192프레임 각각의
#       방위각(camera_pose.zip — 프레임 0 = 0°, 32프레임마다 60°, 모든 영상 같은 방향, 고도 30° 고정)
#       실루엣 중심(cx, cy ∈ [-1, 1])·크기(bbox 긴 변 / 화면) — 흰 배경 영상에서 min(RGB) < 235
#     → 영상 프레임 단위 표(192 × 4)로 저장, fMRI 프레임과의 대응(지연)은 학습 때 정함
#  ② TRELLIS 조건 토큰 정답용 이미지: 영상 프레임 0·32·64·96·128·160 (1차는 0번만 씀, 6뷰는 이후) 원본 1024²
#  ③ 형태 감독 정답 (3D GT, 사용자 10-27 허용): ShapeNetCore v2 model_normalized.obj → 중심·최장변 1로 정규화 [-0.5, 0.5]³
#       → 표면 점 20만 개 → 64³ 점유 격자 (TRELLIS sparse structure와 같은 해상도·범위), np.packbits
# 출력: <out>/targets/<uid>.npz (az_deg, cx, cy, scale: 각 192) / <out>/frames/<uid>/{0..5}.png / <out>/voxels/<uid>.npy (packbits 32,768 B)
# 사용: python -I prep_shape_targets.py --out <폴더> [--workers 10] [--limit N]
import os, io, sys, zipfile, argparse, tempfile
import numpy as np
from multiprocessing import Pool
p = argparse.ArgumentParser()
p.add_argument('--data', default=r'C:\Users\jinsw712\Desktop\Files\EEG\DataSet\MinD-3D++')
p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=10); p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
SH = os.path.join(a.data, 'fMRI-Shape')
IDS = [l.strip() for f in ('core_train_list.txt', 'core_test_list.txt') for l in open(os.path.join(SH, 'annotations', f)) if l.strip()]
if a.limit: IDS = IDS[:a.limit]
VIEW_FR = [0, 32, 64, 96, 128, 160]

def camera_az():
    z = zipfile.ZipFile(os.path.join(SH, 'camera_pose.zip')); az = []
    for i in range(192):
        M = np.load(io.BytesIO(z.read(f'camera_pose/{i:03d}.npy'))); R, t = M[:, :3], M[:, 3]; C = -R.T @ t
        az.append(np.degrees(np.arctan2(C[1], C[0])) % 360)
    return np.array(az, np.float32)

def work(k):
    import cv2, trimesh
    syn, uid = k.split('/')
    done = all(os.path.exists(x) for x in (f'{a.out}/targets/{uid}.npz', f'{a.out}/voxels/{uid}.npy', f'{a.out}/frames/{uid}/5.png'))
    if done: return k, 'skip'
    try:
        # ①② 영상
        z = zipfile.ZipFile(os.path.join(SH, 'stimuli.zip'))
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as t: t.write(z.read(f'stimuli/{k}.mp4')); tmp = t.name
        cap = cv2.VideoCapture(tmp); cx, cy, sc = [], [], []; i = 0
        os.makedirs(f'{a.out}/frames/{uid}', exist_ok=True)
        while True:
            ok, f = cap.read()
            if not ok: break
            m = f.min(2) < 235; ys, xs = np.nonzero(m); H, W = m.shape
            if len(xs):
                x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
                cx.append((x0 + x1) / W - 1); cy.append((y0 + y1) / H - 1); sc.append(max(x1 - x0, y1 - y0) / max(H, W))
            else:
                cx.append(0.); cy.append(0.); sc.append(0.)
            if i in VIEW_FR: cv2.imwrite(f'{a.out}/frames/{uid}/{VIEW_FR.index(i)}.png', f)
            i += 1
        cap.release(); os.remove(tmp)
        if i != 192: return k, f'FAIL frames {i}'
        np.savez(f'{a.out}/targets/{uid}.npz', az_deg=AZ, cx=np.float32(cx), cy=np.float32(cy), scale=np.float32(sc))
        # ③ 복셀
        zm = zipfile.ZipFile(os.path.join(a.data, 'GTmeshes', 'ShapeNetCore', f'{syn}.zip'))
        name = next(n for n in zm.namelist() if n.endswith(f'{uid}/models/model_normalized.obj'))
        mesh = trimesh.load(io.BytesIO(zm.read(name)), file_type='obj', force='mesh', process=False)
        v = np.asarray(mesh.vertices); lo, hi = v.min(0), v.max(0)
        pts = (trimesh.sample.sample_surface(mesh, 200000, seed=0)[0] - (lo + hi) / 2) / (hi - lo).max()
        idx = np.clip(np.floor((pts + 0.5) * 64).astype(int), 0, 63)
        occ = np.zeros((64, 64, 64), bool); occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        np.save(f'{a.out}/voxels/{uid}.npy', np.packbits(occ.ravel()))
        return k, f'ok vox {int(occ.sum())}'
    except Exception as e:
        return k, f'FAIL {repr(e)[:120]}'

AZ = camera_az()
if __name__ == '__main__':
    for d in ('targets', 'voxels', 'frames'): os.makedirs(f'{a.out}/{d}', exist_ok=True)
    print('objects', len(IDS), 'az[0,32,64,96,128,160]', AZ[VIEW_FR].round(1).tolist(), flush=True)
    fails, nv = [], []
    with Pool(a.workers) as pool:
        for c, (k, st) in enumerate(pool.imap_unordered(work, IDS)):
            if st.startswith('FAIL'): fails.append((k, st))
            elif st.startswith('ok'): nv.append(int(st.split()[-1]))
            if (c + 1) % 100 == 0: print(f'{c + 1}/{len(IDS)}', flush=True)
    print('fails', len(fails), fails[:5]); print('voxels occupied mean', np.mean(nv) if nv else '-'); print('PREP_DONE', flush=True)
