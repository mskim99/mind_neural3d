# -*- coding: utf-8 -*-
# 11-24 V-3 (나): 완성된 메시를 목표 세 축 비율에 맞게 축마다 늘이고 줄인 GLB를 조건별로 만든다 (재학습 없음, 평가는 run_eval.sh FE_SKIP_TEX=1로 따로). V-2 v2_rescale.py의 gt3와 같은 절차
#  메시 축 길이 em = 표면 점 8,192개(seed 0)의 축별 범위. 목표 비율 t(세 값, 크기 무관)에 대해 배율 = (t ÷ em) ÷ 그중 최대 (가장 많이 늘어나는 축을 1로)
#  조건: fenc fMRI에서 읽은 비율(고정 인코더 출력, 정답 안 씀) / fraw 같은 읽기를 원 신호로(추가) / predcat 예측 범주의 train 평균 비율(정답 안 씀)
#        / gtcat 정답 범주의 train 평균 비율(진단용) / other 같은 범주의 다른 test 물체 정답 비율(대조, V-2와 같은 물체 seed 0)
#  출력: <out>/<조건>/<uid>/<uid>.glb, <out>/v3.json (물체별 메시 · 정답 축 길이, 조건별 목표 · 배율 · 정답과의 비율 오차)
# 사용: python v3_rescale.py --glb <방향 보정 뒤 glb 폴더> --comp <components.npz> --pred <pred_sub.npz> --catmean <cat_mean.npz> --data_root <fMRI-Shape> --gt_pc <번들 gt_pc/P2> --out <폴더>
import os, json, argparse, numpy as np, trimesh
p = argparse.ArgumentParser()
for k in ('glb', 'comp', 'pred', 'catmean', 'data_root', 'gt_pc', 'out'): p.add_argument('--' + k, required=True)
a = p.parse_args()
TE = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', 'core_test_list.txt')) if l.strip()]; cat = lambda k: k.split('/')[0]; uid = lambda k: k.split('/')[1]
Z = np.load(a.pred, allow_pickle=True); PI = {str(n): i for i, n in enumerate(Z['name'])}; cm = np.load(a.catmean, allow_pickle=True); CM = {str(c): m for c, m in zip(cm['classes'], cm['mean'])}
C = np.load(a.comp, allow_pickle=True); CI = {str(n): i for i, n in enumerate(C['name'])}; CLS = [str(c) for c in C['classes']]; assert set(CLS) == set(CM)
lc = lambda e: np.log(e) - np.log(e).mean()                                  # 축 길이 → 로그 비율
rng = np.random.RandomState(0); by = {}
for k in TE: by.setdefault(cat(k), []).append(k)
EG = {uid(k): (lambda g: g.max(0) - g.min(0))(np.load(f'{a.gt_pc}/{uid(k)}.npy')[0]) for k in TE}
REC = {}; CONDS = ('fenc', 'fraw', 'predcat', 'gtcat', 'other')
for k in TE:
    u = uid(k); i = CI[k] if k in CI else CI[u]; pc = CLS[int(C['cls_pred'][i])]; o = rng.choice([x for x in by[cat(k)] if x != k])
    path = f'{a.glb}/{u}/{u}.glb'; pts = trimesh.sample.sample_surface(trimesh.load(path, force='mesh'), 8192, seed=0)[0]; em = pts.max(0) - pts.min(0); eg = EG[u]
    T = {'fenc': Z['pred_enc'][PI[k]], 'fraw': Z['pred_raw'][PI[k]], 'predcat': CM[pc], 'gtcat': CM[cat(k)], 'other': lc(EG[uid(o)])}
    Fc = {}
    for c in CONDS:
        f = np.exp(T[c]) / em; f = f / f.max(); Fc[c] = [float(v) for v in f]
        sc = trimesh.load(path); sc.apply_transform(np.diag([f[0], f[1], f[2], 1.0])); os.makedirs(f'{a.out}/{c}/{u}', exist_ok=True); sc.export(f'{a.out}/{c}/{u}/{u}.glb')
    REC[u] = {'cat': cat(k), 'pred_cat': pc, 'mesh_ext': em.tolist(), 'gt_ext': eg.tolist(), 'other': uid(o), 'target': {c: [float(v) for v in T[c]] for c in CONDS}, 'factor': Fc,
              'err_mesh': float(np.abs(lc(em) - lc(eg)).mean()), 'err': {c: float(np.abs(T[c] - lc(eg)).mean()) for c in CONDS}}
U = list(REC); hit = np.array([REC[u]['cat'] == REC[u]['pred_cat'] for u in U]); G = np.array([lc(np.array(REC[u]['gt_ext'])) for u in U])
r = lambda x, y: [round(float(np.corrcoef(x[:, d], y[:, d])[0, 1]), 4) for d in range(3)]
def acc(m):
    o = {'n': int(m.sum()), 'mesh': {'corr': r(np.array([lc(np.array(REC[u]['mesh_ext'])) for u in U])[m], G[m]), 'err': round(float(np.mean([REC[u]['err_mesh'] for u, x in zip(U, m) if x])), 4)}}
    for c in CONDS: o[c] = {'corr': r(np.array([REC[u]['target'][c] for u in U])[m], G[m]), 'err': round(float(np.mean([REC[u]['err'][c] for u, x in zip(U, m) if x])), 4)}
    return o
OUT = {'cat_acc': round(float(hit.mean()), 4), 'acc_all': acc(np.ones(len(U), bool)), 'acc_cat_hit': acc(hit), 'acc_cat_miss': acc(~hit), 'objects': REC}
json.dump(OUT, open(f'{a.out}/v3.json', 'w'), ensure_ascii=False, indent=1)
print('V3_RESCALE_DONE', a.out, json.dumps({k: OUT[k] for k in ('cat_acc', 'acc_all')}, ensure_ascii=False), flush=True)
