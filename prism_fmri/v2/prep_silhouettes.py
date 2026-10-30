# PRISM v2 Variational 새 정답 (10-30 사용자 확정 B1): 자극 영상 프레임의 물체 실루엣 32×32
#  fMRI 프레임 k(0~9) ↔ 영상 프레임 round(19.2k) (train_prism_v2.py와 같은 대응). 10프레임 모두 저장, 학습은 2~7만 씀
#  실루엣 = 흰 배경에서 min(RGB) < 235 (prep_shape_targets.py와 같은 기준) → 전체 화면을 32×32로 면적 평균 (0~1 부드러운 값)
#   화면 전체 기준이라 center·scale 정보도 같이 담긴다.
# 출력: <out> (npz) — uids [N], sil uint8 [N, 10, 32, 32] (0~255)
# 사용: python -I prep_silhouettes.py --out <파일.npz> [--workers 10] [--limit N]
import os, sys, zipfile, tempfile, argparse
import numpy as np
from multiprocessing import Pool
p = argparse.ArgumentParser()
p.add_argument('--data', default=r'C:\Users\jinsw712\Desktop\Files\EEG\DataSet\MinD-3D++')
p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=10); p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
SH = os.path.join(a.data, 'fMRI-Shape')
IDS = [l.strip() for f in ('core_train_list.txt', 'core_test_list.txt') for l in open(os.path.join(SH, 'annotations', f)) if l.strip()]
if a.limit: IDS = IDS[:a.limit]
VF = [min(191, round(19.2 * k)) for k in range(10)]


def work(k):
    import cv2
    try:
        z = zipfile.ZipFile(os.path.join(SH, 'stimuli.zip'))
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as t:
            t.write(z.read(f'stimuli/{k}.mp4')); tmp = t.name
        cap = cv2.VideoCapture(tmp); out = {}; i = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            if i in VF:
                m = (f.min(2) < 235).astype(np.float32)
                out[i] = cv2.resize(m, (32, 32), interpolation=cv2.INTER_AREA)
            i += 1
        cap.release(); os.remove(tmp)
        if i != 192: return k, None, f'FAIL frames {i}'
        return k, np.stack([out[v] for v in VF]), 'ok'
    except Exception as e:
        return k, None, f'FAIL {repr(e)[:120]}'


if __name__ == '__main__':
    print('objects', len(IDS), 'video frames', VF, flush=True)
    uids, sils, fails = [], [], []
    with Pool(a.workers) as pool:
        for c, (k, s, st) in enumerate(pool.imap(work, IDS)):
            if s is None: fails.append((k, st))
            else: uids.append(k.split('/')[1]); sils.append(np.round(s * 255).astype(np.uint8))
            if (c + 1) % 200 == 0: print(f'{c + 1}/{len(IDS)}', flush=True)
    sil = np.stack(sils)
    np.savez_compressed(a.out, uids=np.array(uids), sil=sil, video_frames=np.array(VF))
    print('saved', a.out, sil.shape, 'fails', len(fails), fails[:5], 'mean fill', float(sil.mean() / 255), flush=True)
    print('SIL_DONE', flush=True)
