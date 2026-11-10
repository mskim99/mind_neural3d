# 생성물 수평 방향(yaw) 어긋남 확인 (11-11, CPU) — 렌더 지표(R2)는 GT와 같은 카메라로 비교하므로 방향이 틀리면 모양이 맞아도 손해
#  CD · EMD는 공식 adjust_mesh(PCA)로 방향을 맞춘 뒤 재서 영향 거의 없음 → 여기서는 PCA 없이 원 좌표로 봄
#  GT  : 번들 gt_pc/P2/<uid>.npy (GT 메시 표면 → pc_norm, 원 좌표 Y-up) seed 0
#  생성: trimesh.load (GLB = glTF Y-up 그대로, objZ = InstantMesh Z-up → Y-up으로 세움, objY 그대로) → 표면 2048 → pc_norm
#  Y축(위) 기준 0 · 30 · … · 330° 돌려 대칭 평균 최근접 거리(CD) → 가장 작은 각도. 렌더(render.py)도 같은 축 변환이라 이 각도 = 렌더 어긋남
# 사용: python yaw_check.py --pred <폴더> --fmt glb|objZ|objY --gt_pc <번들>/gt_pc/P2 --list <번들>/core_test_list.txt --out <json> [--name 이름]
import os, json, argparse, numpy as np, trimesh
from scipy.spatial import cKDTree
NAME = {'02691156': 'airplane', '02828884': 'bench', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03211117': 'monitor',
        '03636649': 'lamp', '03691459': 'loudspeaker', '04090263': 'rifle', '04256520': 'sofa', '04379243': 'table', '04401088': 'telephone', '04530566': 'boat'}
p = argparse.ArgumentParser()
p.add_argument('--pred', required=True); p.add_argument('--fmt', required=True, choices=['glb', 'objZ', 'objY'])
p.add_argument('--gt_pc', required=True); p.add_argument('--list', required=True); p.add_argument('--out', required=True)
p.add_argument('--name', default=''); p.add_argument('--step', type=int, default=30)
a = p.parse_args()


def norm(x):
    x = x - x.mean(0); return x / (2 * np.sqrt((x ** 2).sum(1)).max())


def cd(x, y):
    return float(cKDTree(y).query(x)[0].mean() + cKDTree(x).query(y)[0].mean())


def roty(t):
    c, s = np.cos(t), np.sin(t); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


ANG = list(range(0, 360, a.step)); rows = []
for l in open(a.list):
    l = l.strip()
    if not l: continue
    syn, uid = l.split('/')
    f = f'{a.pred}/{uid}/{uid}.glb' if a.fmt == 'glb' else f'{a.pred}/{uid}.obj'
    g = f'{a.gt_pc}/{uid}.npy'
    if not (os.path.exists(f) and os.path.exists(g)): continue
    try:
        m = trimesh.load(f, force='mesh'); x = trimesh.sample.sample_surface(m, 2048, seed=0)[0]
    except Exception as e:
        print('LOAD_FAIL', uid, type(e).__name__, flush=True); continue
    if a.fmt == 'objZ': x = x[:, [0, 2, 1]] * np.array([1, 1, -1])   # Z-up → Y-up (Blender obj_import forward Y · up Z와 같은 자세)
    x = norm(x); gt = norm(np.load(g)[0])
    d = [cd(x @ roty(np.deg2rad(t)).T, gt) for t in ANG]; b = int(np.argmin(d))
    rows.append({'uid': uid, 'cls': NAME[syn], 'best': ANG[b], 'cd0': d[0], 'cdbest': d[b], 'ratio': d[b] / d[0], 'cd': d})
R = {'name': a.name, 'n': len(rows), 'step': a.step, 'rows': rows}
mis = [r for r in rows if r['best'] != 0 and r['ratio'] < 0.8]   # 0°보다 20% 넘게 가까운 각도가 있으면 어긋남
R['n_mis'] = len(mis)
print(f'YAW|{a.name} 물체 {len(rows)} | 0°가 최선 {sum(r["best"] == 0 for r in rows)} | 어긋남(다른 각도가 20%+ 가까움) {len(mis)} | CD 0° {np.mean([r["cd0"] for r in rows]):.4f} → 최선 {np.mean([r["cdbest"] for r in rows]):.4f}', flush=True)
for c in sorted(set(r['cls'] for r in rows)):
    rc = [r for r in rows if r['cls'] == c]; mc = [r for r in rc if r in mis]
    print(f'YAW|  {c:12s} {len(mc)}/{len(rc)} 어긋남 | 각도 {sorted(r["best"] for r in mc)}', flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(R, open(a.out, 'w'), indent=1)
print('YAW_DONE', a.out, flush=True)
