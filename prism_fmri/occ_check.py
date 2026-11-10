# B안 3단계 (11-10): 점유 64³ 기준 검사 — 생성 점유 vs 학습 점유
#  복사 검사: 가장 가까운 학습 물체 3D IoU (생성 / 학습에서 뺀 실제 검증 물체)
#  다양성: 생성끼리 평균 IoU (실제 검증끼리와 비교)
#  범주: 가장 가까운 학습 물체의 ShapeNet 범주 분포 (쏠림 확인)
#  그림: 점유 점을 방위 45° · 고도 30°로 투영한 깊이 그림 (생성 32 + 실제 검증 16)
# 사용: python occ_check.py --train_occ <ss>/occ --train_objs train_objs.txt --val_objs val_objs.txt --gen_occ <gen>/occ --out <폴더>
import os, json, glob, argparse, numpy as np, torch
from collections import Counter
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument('--train_occ', required=True); p.add_argument('--train_objs', required=True); p.add_argument('--val_objs', required=True)
p.add_argument('--gen_occ', required=True); p.add_argument('--out', required=True); p.add_argument('--tag', default='')
a = p.parse_args(); dev = 'cuda'; N3 = 64 ** 3
CAT = {'02691156': 'airplane', '02828884': 'bench', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03211117': 'monitor', '03636649': 'lamp',
       '03691459': 'speaker', '04090263': 'rifle', '04256520': 'sofa', '04379243': 'table', '04401088': 'telephone', '04530566': 'watercraft'}
ld = lambda f: np.unpackbits(np.load(f))[:N3].astype(bool)
tr = [l.strip() for l in open(a.train_objs) if l.strip() and os.path.exists(f'{a.train_occ}/{l.strip()}.npy')]
va = [l.strip() for l in open(a.val_objs) if l.strip() and os.path.exists(f'{a.train_occ}/{l.strip()}.npy')]
va = [va[i] for i in sorted(np.random.RandomState(0).choice(len(va), min(200, len(va)), replace=False))]   # 목록이 범주 순이라 무작위로
gf = sorted(glob.glob(f'{a.gen_occ}/*.npy')); gn = [os.path.basename(f)[:-4] for f in gf]
T = torch.from_numpy(np.stack([ld(f'{a.train_occ}/{u}.npy') for u in tr])).to(dev).half()
Ts = T.float().sum(1)


def nearest(X):                                        # X [n, N3] bool → (최대 IoU, 학습 인덱스)
    X = torch.from_numpy(X).to(dev).half(); Xs = X.float().sum(1); best, arg = torch.zeros(len(X), device=dev), torch.zeros(len(X), dtype=torch.long, device=dev)
    for b in range(0, len(T), 1500):
        I = (X @ T[b:b + 1500].T).float(); U = Xs[:, None] + Ts[None, b:b + 1500] - I; iou = I / U.clamp_min(1)
        m, j = iou.max(1); up = m > best; best[up] = m[up]; arg[up] = j[up] + b
    return best.cpu().numpy(), arg.cpu().numpy()


def pair_iou(X):
    X = torch.from_numpy(X).to(dev).float(); s = X.sum(1); I = X @ X.T; iou = I / (s[:, None] + s[None] - I).clamp_min(1)
    n = len(X); return float((iou.sum() - iou.diag().sum()) / (n * (n - 1)))


G = np.stack([ld(f) for f in gf]); V = np.stack([ld(f'{a.train_occ}/{u}.npy') for u in va]); V16 = np.random.RandomState(1).permutation(len(V))
gi, ga = nearest(G); vi, vaa = nearest(V)
gcat = Counter(CAT.get(tr[j].split('_')[0], '?') for j in ga); vcat = Counter(CAT.get(u.split('_')[0], '?') for u in va)
R = {'tag': a.tag, 'gen_n': len(G), 'gen_nn_iou': float(gi.mean()), 'val_nn_iou': float(vi.mean()), 'gen_pair_iou': pair_iou(G[:64]), 'val_pair_iou': pair_iou(V[V16[:64]]),
     'gen_occ_mean': float(G.sum(1).mean()), 'val_occ_mean': float(V.sum(1).mean()), 'gen_empty': int((G.sum(1) < 50).sum()),
     'gen_nn_cat': dict(gcat.most_common()), 'val_cat': dict(vcat.most_common()), 'gen_nn_iou_each': gi.round(3).tolist(), 'gen_nn_uid': [tr[j] for j in ga]}
os.makedirs(a.out, exist_ok=True); json.dump(R, open(f'{a.out}/occ_check.json', 'w'), indent=1)
print(f"OCC [{a.tag}] 가장 가까운 학습 IoU: 생성 {R['gen_nn_iou']:.3f} vs 실제 검증 {R['val_nn_iou']:.3f} | 서로 IoU: 생성 {R['gen_pair_iou']:.3f} vs 실제 {R['val_pair_iou']:.3f} | 점유 수 {R['gen_occ_mean']:.0f} vs {R['val_occ_mean']:.0f} | 빈 것 {R['gen_empty']}", flush=True)
print('  가장 가까운 범주(생성):', dict(gcat.most_common()), flush=True)


def view(o, S=160):                                    # 점유 → 깊이 그림 (방위 45, 고도 30)
    c = np.argwhere(o.reshape(64, 64, 64)).astype(np.float32) - 31.5
    if len(c) == 0: return Image.new('RGB', (S, S), 'white')
    az, el = np.radians(45), np.radians(30)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]]); Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    q = c @ Rz.T @ Rx.T; W = 64 * 1.75                 # 회전하면 모서리가 ±32√3까지 나감
    u = np.clip(((q[:, 0] / W + 0.5) * (S - 1)).astype(int), 0, S - 1); v = np.clip(((0.5 - q[:, 2] / W) * (S - 1)).astype(int), 0, S - 1); d = q[:, 1]
    img = np.full((S, S), np.inf); o_ = np.argsort(-d); img[v[o_], u[o_]] = d[o_]      # 가까운 점(작은 d)이 마지막에 덮음
    out = np.full((S, S, 3), 255, np.uint8); m = np.isfinite(img)
    sh = (img[m] - img[m].min()) / max(1e-6, np.ptp(img[m])); out[m] = np.stack([60 + 150 * sh] * 3, -1).astype(np.uint8)
    return Image.fromarray(out)


S = 160; cv = Image.new('RGB', (S * 16, 30 + S * 3 + 20), 'white'); d = ImageDraw.Draw(cv)
d.text((4, 4), f'{a.tag}: rows 1-2 = prior samples (number = nearest-train IoU, nearest category), row 3 = real held-out', fill='black')
for i in range(min(32, len(G))):
    r, c = divmod(i, 16); cv.paste(view(G[i]), (c * S, 30 + r * S)); d.text((c * S + 3, 30 + r * S + 3), f'{gi[i]:.2f} {CAT.get(tr[ga[i]].split("_")[0], "?")}', fill='blue')
for k, i in enumerate(V16[:16]):
    i = int(i); cv.paste(view(V[i]), (k * S, 30 + 2 * S + 20)); d.text((k * S + 3, 30 + 2 * S + 23), f'{vi[i]:.2f} {CAT.get(va[i].split("_")[0], "?")}', fill='green')
cv.save(f'{a.out}/occ_check.png'); print('MONTAGE', f'{a.out}/occ_check.png')
