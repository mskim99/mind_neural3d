# GT 없이 고른 방향(trellis_tools yawsel)이 GT로 찾은 최선 방향(yaw_check)과 얼마나 맞는지 (11-11, CPU)
#  yaw_check json의 물체별 cd[각도](30° 간격)를 그대로 써서, 고른 각도로 돌렸을 때 GT 점구름 거리를 바로 계산 (다시 렌더 안 함)
#  부호 확인용으로 반대 방향(+φ)도 같이 냄 — 정해 둔 −φ가 확실히 더 좋아야 축 · 부호가 맞은 것
# 사용: python yaw_compare.py --yaw <yaw_check json> --sel <yawsel.json> --name 이름
import json, argparse, numpy as np
p = argparse.ArgumentParser(); p.add_argument('--yaw', required=True); p.add_argument('--sel', required=True); p.add_argument('--name', default='')
a = p.parse_args()
Y = json.load(open(a.yaw)); S = json.load(open(a.sel)); st = Y['step']; n_ang = 360 // st
rows = [r for r in Y['rows'] if r['uid'] in S]
idx = lambda deg: int(round(deg / st)) % n_ang
mis = lambda r: r['best'] != 0 and r['ratio'] < 0.8


def summary(sign):
    out = []
    for r in rows:
        t = (sign * S[r['uid']]['phi']) % 360; out.append((r, t, r['cd'][idx(t)]))
    return out


for lab, sign in (('정한 방향 −φ', -1), ('반대 방향 +φ (부호 확인)', 1)):
    o = summary(sign)
    cd0 = np.mean([r['cd0'] for r, _, _ in o]); cds = np.mean([c for _, _, c in o]); cdb = np.mean([r['cdbest'] for r, _, _ in o])
    same = sum(idx(t) == idx(r['best']) for r, t, _ in o)
    near = sum(min((idx(t) - idx(r['best'])) % n_ang, (idx(r['best']) - idx(t)) % n_ang) <= 1 for r, t, _ in o)
    m = [(r, t, c) for r, t, c in o if mis(r)]; ok = [(r, t, c) for r, t, c in o if not mis(r)]
    fixed = sum(c < 0.8 * r['cd0'] for r, t, c in m); broke = sum(c > 1.25 * r['cd0'] for r, t, c in ok)
    print(f'YAWCMP|{a.name} {lab} | 물체 {len(o)} | 최선 각도와 같음 {same} · ±{st}° 안 {near} | CD 0° {cd0:.4f} → 고른 각도 {cds:.4f} (GT 최선 {cdb:.4f}) '
          f'| 어긋난 {len(m)}개 중 고침 {fixed} | 맞던 {len(ok)}개 중 망침 {broke}', flush=True)
    if sign == -1:
        for c in sorted(set(r['cls'] for r in rows)):
            oc = [(r, t, x) for r, t, x in o if r['cls'] == c]
            print(f'YAWCMP|  {c:12s} 어긋남 {sum(mis(r) for r, _, _ in oc)} 고침 {sum(mis(r) and x < 0.8 * r["cd0"] for r, _, x in oc)} 망침 {sum((not mis(r)) and x > 1.25 * r["cd0"] for r, _, x in oc)} '
                  f'| 고른 {sorted(int(t) for _, t, _ in oc)} | 최선 {sorted(r["best"] for r, _, _ in oc)}', flush=True)
print('YAWCMP_DONE', flush=True)
