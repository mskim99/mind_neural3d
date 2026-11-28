# 11-28 F-1: Statistic Regressor 정답 — 질감 통계 12개(v9_feat tex_ob) → train PCA PC1 · PC2, 정답 범주 train 평균 뺌, train 표준편차로 나눔 (v10_head.py와 같은 정의 · 같은 부호)
#  PCA · 평균 · 표준편차는 train 1,300개로만. 출력 npz: keys [N] (train + test), app [N, 2], ok [N], Vt [2, 12], sdt [12], sdp [2], cm [범주, 2], cl
# 사용: python prep_app_target.py --data_root $DR --v9 $W/prism_v9/v9_feat.npz --v5 $W/prism_v9/v5_feat.npz --out $TGT/app_target.npz
import os, argparse, numpy as np
p = argparse.ArgumentParser()
for k in ('data_root', 'v9', 'v5', 'out'): p.add_argument('--' + k, required=True)
a = p.parse_args()
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
cat = lambda k: k.split('/')[0]
Z9 = np.load(a.v9); Z5 = np.load(a.v5); K9 = {str(k): i for i, k in enumerate(Z9['keys'])}; K5 = {str(k): i for i, k in enumerate(Z5['keys'])}
ok = set(K9) & set(K5); TR = [k for k in lst('train') if k in ok]; ALL = lst('train') + lst('test'); CL = sorted({cat(k) for k in TR})
T_all = Z9['tex_ob'].astype(np.float64); ttr = T_all[[K9[k] for k in TR]]
cmt = {c: ttr[[i for i, k in enumerate(TR) if cat(k) == c]].mean(0) for c in CL}
rt = ttr - np.stack([cmt[cat(k)] for k in TR]); sdt = rt.std(0); _, St, Vt = np.linalg.svd(rt / sdt, full_matrices=False)
BH = Z5['feat'][:, list(Z5['names']).index('band_hi')].astype(np.float64)
pc = lambda ks: (T_all[[K9[k] for k in ks]] / sdt) @ Vt[:2].T
ptr = pc(TR); cmp_ = {c: ptr[[i for i, k in enumerate(TR) if cat(k) == c]].mean(0) for c in CL}
rtr = ptr - np.stack([cmp_[cat(k)] for k in TR])
bh = np.array([BH[K5[k]] for k in TR]); bh = bh - np.array([np.mean([BH[K5[j]] for j in TR if cat(j) == cat(k)]) for k in TR])
sgn = np.sign(np.corrcoef(rtr[:, 0], bh)[0, 1]); Vt[0] *= sgn; rtr[:, 0] *= sgn; cmp_ = {c: v * np.array([sgn, 1]) for c, v in cmp_.items()}; sdp = rtr.std(0)
app = np.zeros((len(ALL), 2), np.float32); okv = np.zeros(len(ALL), bool)
for i, k in enumerate(ALL):
    if k in K9 and cat(k) in cmp_: app[i] = ((T_all[K9[k]] / sdt) @ Vt[:2].T - cmp_[cat(k)]) / sdp; okv[i] = True
np.savez(a.out, keys=np.array(ALL), app=app, ok=okv, Vt=Vt[:2], sdt=sdt, sdp=sdp, cm=np.stack([cmp_[c] for c in CL]), cl=np.array(CL))
print(f'APP_TARGET {okv.sum()} / {len(ALL)} · 설명 분산 {np.round(St ** 2 / (St ** 2).sum(), 3)[:3].tolist()} · train sd {np.round(app[:len(TR)][okv[:len(TR)]].std(0), 3).tolist()} → {a.out}', flush=True)
