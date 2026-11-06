# 예측 토큰 재조정 스윕 + 범주 지표 (11-07 색 실험 1, 추론만 · 재학습 없음, MinD-3D++ venv, GPU)
#  배경: 예측 편차(토큰 − train 평균 토큰 M)의 크기가 정답의 0.7~0.76배 → 평균 쪽으로 수축해 회색이 나오는지 가리는 진단
#  변형 (M = train 정답 토큰 평균, LN = 토큰마다 1024차원 affine 없는 layer_norm, TRELLIS get_cond 마지막 처리와 같음)
#   ln        : LN(pred)
#   ln_s<스케일>: LN(M + s·(pred − M))
#   s<스케일>  : M + s·(pred − M) — 진단 수치만 (토큰 저장 안 함, ×1.41은 11-05 정성 진단에서 효과 없어 버림)
#  출력 <out>/<변형>/<uid>.npy (fp16, trellis_tools gen/gscheck에 바로 넣음), <out>/renorm_stats.json
#  진단 (test 정답과 비교): 편차 방향 cos, 편차 크기 비, 토큰별 표준편차(정답은 LN이라 ≈1)
#  범주 (components.npz의 cls_score): top-1 · top-3 정확도, kNN 이웃 범주 일치 — 거르지 않음 / 예측 top-3 범주 안에서만 / top-1 범주 안에서만
#   (kNN = 예측 편차와 cos가 가장 큰 train 물체 k개. 실험 2 검색 조건의 사전 확인용, 토큰은 저장 안 함)
# 사용: python renorm.py --pred <예측 tokens 폴더> --comp <components.npz> --tgt <shape_targets> --data_root <fMRI-Shape> --out <폴더> [--scales 1.37,1.6] [--knn 1,3,5]
import os, json, argparse, numpy as np, torch
import torch.nn.functional as F
p = argparse.ArgumentParser()
p.add_argument('--pred', required=True); p.add_argument('--comp', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--data_root', required=True); p.add_argument('--out', required=True)
p.add_argument('--scales', default='1.37,1.6'); p.add_argument('--knn', default='1,3,5')
a = p.parse_args()
ids = lambda sp: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
uid = lambda k: k.split('/')[1]; cat = lambda k: k.split('/')[0]
tr = [k for k in ids('train') if os.path.exists(f'{a.tgt}/dino/{uid(k)}.npy')]; te = [k for k in ids('test') if os.path.exists(f'{a.pred}/{uid(k)}.npy')]
ld = lambda f: torch.from_numpy(np.load(f).astype(np.float32)).flatten()
X = torch.stack([ld(f'{a.tgt}/dino/{uid(k)}.npy') for k in tr]).cuda()           # [1300, D] train 정답 토큰
M = X.mean(0); X -= M                                                              # train 편차
P = torch.stack([ld(f'{a.pred}/{uid(k)}.npy') for k in te]).cuda()                 # [104, D] 예측 토큰
G = torch.stack([ld(f'{a.tgt}/dino/{uid(k)}.npy') for k in te]).cuda() - M         # test 정답 편차 (진단만)
print(f'train {len(tr)} test {len(te)} D {X.shape[1]}', flush=True)
nrm = lambda v: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
ln = lambda t: F.layer_norm(t.view(len(t), 1374, 1024), (1024,)).flatten(1)
out = {}


def diag(name, Q):                                                                 # Q: 토큰 (편차 아님)
    D = Q - M; dc = (nrm(D) * nrm(G)).sum(-1); mr = D.norm(dim=-1) / G.norm(dim=-1)
    sd = Q.view(len(Q), 1374, 1024).std(-1).mean()
    out[name] = dict(dev_cos=float(dc.mean()), mag_ratio=float(mr.mean()), tok_std=float(sd))
    print(f'TOK {name:<8} 편차 방향 cos {dc.mean():.3f} · 크기 비 {mr.mean():.2f} · 토큰 표준편차 {sd:.3f}', flush=True)


def save(name, Q):
    os.makedirs(f'{a.out}/{name}', exist_ok=True)
    for k, q in zip(te, Q.cpu()):
        np.save(f'{a.out}/{name}/{uid(k)}.npy', q.view(1374, 1024).half().numpy())


diag('gt', G + M); diag('pred', P)
V = {'ln': ln(P)}
for s in map(float, a.scales.split(',')):
    t = f'{round(s * 100)}'; V[f's{t}'] = M + s * (P - M); V[f'ln_s{t}'] = ln(V[f's{t}'])
for n, Q in V.items():
    diag(n, Q)
    if n.startswith('ln'): save(n, Q)
del V
# 범주: top-1 · top-3, kNN 이웃 범주 일치 (거르지 않음 / top-3 / top-1 안에서)
c = np.load(a.comp); SYN = [str(x) for x in c['classes']]; row = {uid(str(n)): i for i, n in enumerate(c['name'])}
S = torch.from_numpy(c['cls_score'][[row[uid(k)] for k in te]]).float()           # [104, 13]
y = torch.tensor([SYN.index(cat(k)) for k in te]); top3 = S.topk(3, dim=1).indices
out['cat'] = dict(top1=float((top3[:, 0] == y).float().mean()), top3=float((top3 == y[:, None]).any(1).float().mean()))
print(f"CAT top-1 {out['cat']['top1']:.3f} · top-3 {out['cat']['top3']:.3f} (test {len(te)})", flush=True)
trc = torch.tensor([SYN.index(cat(k)) for k in tr]).cuda()
C = nrm(P - M) @ nrm(X).T                                                          # [104, 1300] 편차 cos
gate = {'all': torch.ones_like(C, dtype=torch.bool),
        'top3': (trc[None, :, None] == top3.cuda()[:, None, :]).any(-1),
        'top1': trc[None, :] == top3[:, :1].cuda()}
yc = y.cuda()
for g, m in gate.items():
    for k in map(int, a.knn.split(',')):
        i = C.masked_fill(~m, -2).topk(k, dim=1).indices
        hit = float((trc[i] == yc[:, None]).float().mean())
        out['cat'][f'nn{k}_{g}'] = hit
    print(f'KNN {g:<4} 이웃 범주 일치 ' + ' · '.join(f"k{k} {out['cat'][f'nn{k}_{g}']:.3f}" for k in map(int, a.knn.split(','))), flush=True)
json.dump(out, open(f'{a.out}/renorm_stats.json', 'w'), indent=1, ensure_ascii=False); print('RENORM_TOK_DONE', flush=True)
