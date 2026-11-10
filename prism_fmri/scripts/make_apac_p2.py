# APAC 220물체 GT 점구름 P2 (11-10, 로컬 CPU): 공정평가 번들 make_gt_pc.py의 P2 계산을 그대로 옮긴 것 (번들 스크립트는 ONet 점구름이 꼭 필요해 APAC에 못 씀)
#   P2: GT 메시(ShapeNetCore v2 model_normalized.obj) 표면 8192 (seed s) → pc_norm → 무작위 2048 (default_rng(s), 중복 허용), s = 0~4 → <out>/<uid>.npy (5, 2048, 3) float32
#   uid는 목록 그대로(30·31·32자). ShapeNet 폴더 이름이 앞자리 0으로 채워진 경우 lstrip('0')으로 맞춤
# 사용: python make_apac_p2.py --list apac_list.txt --zips <ShapeNetCore zip 폴더> --pc_norm_from <번들 scripts 폴더> --out <gt_pc/P2>
import os, sys, io, argparse, zipfile, numpy as np, trimesh
ap = argparse.ArgumentParser()
ap.add_argument('--list', required=True); ap.add_argument('--zips', required=True); ap.add_argument('--pc_norm_from', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
sys.path.insert(0, a.pc_norm_from); from official_pc import pc_norm
os.makedirs(a.out, exist_ok=True)
keys = [l.strip() for l in open(a.list) if l.strip()]
Z, done, miss = {}, 0, []
for k in keys:
    syn, uid = k.split('/'); f = f'{a.out}/{uid}.npy'
    if os.path.exists(f): done += 1; continue
    if syn not in Z:
        z = zipfile.ZipFile(f'{a.zips}/{syn}.zip'); Z[syn] = (z, {n.split('/')[1].lstrip('0'): n for n in z.namelist() if n.endswith('models/model_normalized.obj')})
    z, idx = Z[syn]; n = idx.get(uid.lstrip('0'))
    if n is None: miss.append(k); continue
    m = trimesh.load(io.BytesIO(z.read(n)), file_type='obj', force='mesh')
    out = []
    for s in range(5):
        g = pc_norm(trimesh.sample.sample_surface(m, 8192, seed=s)[0])
        out.append(g[np.random.default_rng(s).choice(len(g), 2048, replace=True)])
    np.save(f, np.stack(out).astype(np.float32)); done += 1
    if done % 20 == 0: print('P2', done, '/', len(keys), flush=True)
print('APAC_P2_DONE', done, '/', len(keys), '없음', miss, flush=True)
