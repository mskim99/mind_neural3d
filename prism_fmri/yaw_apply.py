# yawsel이 고른 수평 회전을 GLB에 적용 (11-11, CPU) — GT를 쓰지 않은 각도(yawsel.json의 yaw_fix)만 씀
#  회전은 glTF 위쪽 축(Y) 기준, yaw_check.py의 roty와 같은 방향 → 정점에 직접 구움 (텍스처 · 재질 그대로)
#  yaw_fix = 0인 물체는 파일을 그대로 복사 (평가 입력이 바이트까지 같음)
# 사용: python yaw_apply.py --glb <폴더(<uid>/<uid>.glb)> --sel <yawsel.json> --out <폴더>
import os, json, shutil, argparse, collections, numpy as np, trimesh
p = argparse.ArgumentParser(); p.add_argument('--glb', required=True); p.add_argument('--sel', required=True); p.add_argument('--out', required=True)
a = p.parse_args()
S = json.load(open(a.sel)); cnt = collections.Counter(); miss = 0
for uid in sorted(os.listdir(a.glb)):
    f = f'{a.glb}/{uid}/{uid}.glb'
    if not os.path.exists(f): continue
    o = f'{a.out}/{uid}/{uid}.glb'; os.makedirs(os.path.dirname(o), exist_ok=True)
    if uid not in S: miss += 1
    t = float(S.get(uid, {}).get('yaw_fix', 0)) % 360; cnt[int(t)] += 1
    if t == 0: shutil.copyfile(f, o); continue
    c, s = np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t))
    M = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    sc = trimesh.load(f)
    gs = sc.dump() if isinstance(sc, trimesh.Scene) else [sc]   # 노드 변환을 정점에 적용한 메시 목록 (재질 유지)
    for g in gs: g.apply_transform(M)
    trimesh.Scene(gs).export(o)
print(f'YAWAPPLY_DONE {sum(cnt.values())}개 → {a.out} | 각도별 {sorted(cnt.items())} | yawsel에 없어 그대로 둔 것 {miss}', flush=True)
