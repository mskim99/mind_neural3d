# -*- coding: utf-8 -*-
# 11-20 V-2: 완성된 메시의 세로 · 가로 비율을 목표 폭 · 높이에 맞게 조정한 GLB를 조건별로 만든다 (재학습 없음, 평가는 평가 번들 run_eval.sh FE_SKIP_TEX=1로 따로)
#  메시 폭 · 높이 = 평가 렌더 render_R2 6개 시점의 실루엣 bbox(화면 비율) 평균. 목표 폭 · 높이 = 자극 영상 6프레임 실루엣(view_sil.npz [cx, cy, w, h, area]) 평균과 같은 단위
#   (정답 104개에서 평가 렌더와 영상의 폭 · 높이 상관 0.99 · 평균 0.517 / 0.504 · 0.533 / 0.526 확인 → 변환 없이 씀)
#  조정: 세로(y)를 목표 높이 ÷ 메시 높이, 가로 두 축(x · z)을 목표 폭 ÷ 메시 폭 배. 평가가 중심 · 최대 반지름으로 정규화하므로 비율만 남음
#  조건: same 그대로(다시 저장만) / gt3 정답 점구름의 세 축 길이로 축마다 / gtsil 정답 실루엣 / vpred Variational 예측(표준화 값 × train 표준편차 + 예측 범주의 train 평균)
#        / catmean 예측 범주의 train 평균 / other 같은 범주의 다른 test 물체 정답(seed 0)
#  출력: <out>/<조건>/<uid>/<uid>.glb, <out>/v2.json (물체별 폭 · 높이 · 배율, Variational 예측과 정답의 상관)
# 사용: python v2_rescale.py --glb <방향 보정 뒤 glb 폴더> --render <render_R2> --comp <components.npz> --sil <view_sil.npz> --data_root <fMRI-Shape> --gt_pc <번들 gt_pc/P2> --out <폴더>
import os, json, argparse, numpy as np, trimesh
from PIL import Image
p = argparse.ArgumentParser()
for k in ('glb', 'render', 'comp', 'sil', 'data_root', 'gt_pc', 'out'): p.add_argument('--' + k, required=True)
a = p.parse_args()
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
TR, TE = lst('train'), lst('test'); z = np.load(a.sil); KI = {str(k): i for i, k in enumerate(z['keys'])}; S = z['sil'].astype(np.float64)
ok = ~np.isnan(S).any((1, 2)); So = np.nan_to_num(S).mean(1)[:, 1:]                       # 물체 평균 [cy, w, h, area]
cat = lambda k: k.split('/')[0]; uid = lambda k: k.split('/')[1]
tri = [KI[k] for k in TR if k in KI and ok[KI[k]]]; CM = {}
for c in sorted({cat(k) for k in TR}): CM[c] = So[[i for i in tri if cat(str(z['keys'][i])) == c]].mean(0)
res = So[tri] - np.stack([CM[cat(str(z['keys'][i]))] for i in tri]); SSD = res.std(0) + 1e-8      # build_view_targets.py shape와 같은 통계
C = np.load(a.comp, allow_pickle=True); CI = {str(n): i for i, n in enumerate(C['name'])}; CLS = [str(c) for c in C['classes']]
assert set(CLS) == set(CM), (CLS, list(CM))


def wh(d):                                                                               # render_R2/<uid>/0..5.png → 폭 · 높이 평균 (화면 비율)
    o = []
    for v in range(6):
        im = np.array(Image.open(f'{d}/{v}.png')); m = im[..., 3] > 10 if im.shape[-1] == 4 else im[..., :3].min(-1) < 235
        ys, xs = np.nonzero(m); o.append(((xs.max() - xs.min() + 1) / im.shape[1], (ys.max() - ys.min() + 1) / im.shape[0]))
    return np.mean(o, 0)


rng = np.random.RandomState(0); by = {}
for k in TE: by.setdefault(cat(k), []).append(k)
REC = {}; CONDS = ('same', 'gt3', 'gtsil', 'vpred', 'catmean', 'other')
for k in TE:
    u = uid(k); i = CI[k] if k in CI else CI[u]; pc = CLS[int(C['cls_pred'][i])]; vt = C['vt'][i].astype(np.float64)
    mw, mh = wh(f'{a.render}/{u}'); gw, gh = So[KI[k]][1], So[KI[k]][2]
    o = rng.choice([x for x in by[cat(k)] if x != k]); T = {'gtsil': (gw, gh), 'vpred': (vt[1] * SSD[1] + CM[pc][1], vt[2] * SSD[2] + CM[pc][2]), 'catmean': (CM[pc][1], CM[pc][2]), 'other': (So[KI[o]][1], So[KI[o]][2])}
    path = f'{a.glb}/{u}/{u}.glb'; pts = trimesh.sample.sample_surface(trimesh.load(path, force='mesh'), 8192, seed=0)[0]; em = pts.max(0) - pts.min(0)
    g = np.load(f'{a.gt_pc}/{u}.npy')[0]; eg = g.max(0) - g.min(0)
    F = {'same': (1.0, 1.0, 1.0), 'gt3': tuple(float(v) for v in (eg / em) / (eg / em).max())}
    for c, (tw, th) in T.items(): b, v_ = max(tw, 1e-3) / mw, max(th, 1e-3) / mh; n = max(b, v_); F[c] = (b / n, v_ / n, b / n)   # (x, y, z) 배율, 큰 쪽을 1로
    for c in CONDS:
        sc = trimesh.load(path); Mx = np.diag([F[c][0], F[c][1], F[c][2], 1.0])
        sc.apply_transform(Mx); os.makedirs(f'{a.out}/{c}/{u}', exist_ok=True); sc.export(f'{a.out}/{c}/{u}/{u}.glb')
    REC[u] = {'cat': cat(k), 'pred_cat': pc, 'mesh_wh': [float(mw), float(mh)], 'gt_wh': [float(gw), float(gh)], 'other': uid(o), 'vt': vt.tolist(),
              'target': {c: [float(t[0]), float(t[1])] for c, t in T.items()}, 'factor': {c: list(F[c]) for c in CONDS}}
# Variational 예측이 얼마나 맞나: 범주 평균을 뺀 값(정답은 정답 범주의 train 평균을 뺌, 예측은 표준화 값 × 표준편차) · 절대값 · 세로/가로 비
U = list(REC); hit = np.array([REC[u]['cat'] == REC[u]['pred_cat'] for u in U])
gres = np.array([[REC[u]['gt_wh'][0] - CM[REC[u]['cat']][1], REC[u]['gt_wh'][1] - CM[REC[u]['cat']][2]] for u in U]); pres = np.array([[REC[u]['vt'][1] * SSD[1], REC[u]['vt'][2] * SSD[2]] for u in U])
gabs = np.array([REC[u]['gt_wh'] for u in U]); lr = lambda x: np.log(x[:, 1] / x[:, 0]); r = lambda x, y: round(float(np.corrcoef(x, y)[0, 1]), 4) if len(x) > 2 else None
def acc(m):
    o = {'n': int(m.sum()), 'w_res': r(pres[m, 0], gres[m, 0]), 'h_res': r(pres[m, 1], gres[m, 1])}
    for c in ('vpred', 'catmean', 'other'):
        t = np.array([REC[u]['target'][c] for u in U]); o[c] = {'w': r(t[m, 0], gabs[m, 0]), 'h': r(t[m, 1], gabs[m, 1]), 'h_over_w': r(lr(t[m]), lr(gabs[m]))}
    o['mesh'] = {'h_over_w': r(lr(np.array([REC[u]['mesh_wh'] for u in U])[m]), lr(gabs[m]))}; return o
OUT = {'acc_all': acc(np.ones(len(U), bool)), 'acc_cat_hit': acc(hit), 'acc_cat_miss': acc(~hit), 'cat_acc': round(float(hit.mean()), 4), 'ssd': SSD.tolist(), 'objects': REC}
json.dump(OUT, open(f'{a.out}/v2.json', 'w'), ensure_ascii=False, indent=1)
print('V2_RESCALE_DONE', a.out, json.dumps({k: OUT[k] for k in ('cat_acc', 'acc_all', 'acc_cat_hit', 'acc_cat_miss')}, ensure_ascii=False), flush=True)
