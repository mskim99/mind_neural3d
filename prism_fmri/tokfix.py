# 예측 토큰을 train 정답 토큰 공간으로 옮기기 (11-06, 추론만 · 재학습 없음, MinD-3D++ venv, GPU)
#  쓰는 정답: train 물체의 정답 DINO 토큰만 (<tgt>/dino/<uid>.npy, 1300개). test 정답은 아래 '진단' 통계에만 씀
#  편차 = 토큰 − train 평균 토큰 M (1374×1024를 펼친 벡터)
#   knn<k> : 예측 편차와 cos가 가장 큰 train 물체 k개 → 정답 토큰을 cos 가중 평균 (k=1이면 가장 가까운 train 물체 그대로)
#   pca<p> : train 편차의 주성분 p개 공간에 예측 편차를 투영 → M + 투영
#  출력 <out>/<변형>/<uid>.npy (fp16, trellis_tools gen/gscheck에 바로 넣음), <out>/tokfix_stats.json
#  진단 (test 정답과 비교): 편차 방향 cos, 편차 크기 비, kNN 이웃 범주가 맞은 비율
# 사용: python tokfix.py --pred <예측 tokens 폴더> --tgt <shape_targets> --data_root <fMRI-Shape> --out <폴더> [--knn 1,3,5] [--pca 16,64,256]
import os, json, argparse, numpy as np, torch
p = argparse.ArgumentParser()
p.add_argument('--pred', required=True); p.add_argument('--tgt', required=True); p.add_argument('--data_root', required=True); p.add_argument('--out', required=True)
p.add_argument('--knn', default='1,3,5'); p.add_argument('--pca', default='16,64,256')
a = p.parse_args()
ids = lambda sp: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
tr = [k for k in ids('train') if os.path.exists(f'{a.tgt}/dino/{k.split("/")[1]}.npy')]; te = [k for k in ids('test') if os.path.exists(f'{a.pred}/{k.split("/")[1]}.npy')]
uid = lambda k: k.split('/')[1]; cat = lambda k: k.split('/')[0]
ld = lambda f: torch.from_numpy(np.load(f).astype(np.float32)).flatten()
X = torch.stack([ld(f'{a.tgt}/dino/{uid(k)}.npy') for k in tr]).cuda()           # [1300, D] 정답 토큰
M = X.mean(0); X -= M                                                              # 편차
P = torch.stack([ld(f'{a.pred}/{uid(k)}.npy') for k in te]).cuda() - M             # [104, D] 예측 편차
G = torch.stack([ld(f'{a.tgt}/dino/{uid(k)}.npy') for k in te]).cuda() - M         # test 정답 편차 (진단만)
print(f'train {len(tr)} test {len(te)} D {X.shape[1]}', flush=True)
nrm = lambda v: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
C = nrm(P) @ nrm(X).T                                                              # [104, 1300] 편차 cos
out = {}


def diag(name, Q):
    dc = (nrm(Q) * nrm(G)).sum(-1); mr = Q.norm(dim=-1) / G.norm(dim=-1)
    out.setdefault(name, {}).update(dev_cos=float(dc.mean()), mag_ratio=float(mr.mean()))
    print(f'TOK {name:<7} 편차 방향 cos {dc.mean():.3f} · 크기 비 {mr.mean():.2f}' + (f" · 이웃 범주 일치 {out[name]['nn_cat']:.3f}" if 'nn_cat' in out[name] else ''), flush=True)


def save(name, Q):
    os.makedirs(f'{a.out}/{name}', exist_ok=True)
    for k, q in zip(te, (Q + M).cpu()):
        np.save(f'{a.out}/{name}/{uid(k)}.npy', q.view(1374, 1024).half().numpy())


diag('pred', P)
for k in map(int, a.knn.split(',')):
    v, i = C.topk(k, dim=1); w = v.clamp_min(0) + 1e-6; w = w / w.sum(1, keepdim=True)
    Q = torch.einsum('nk,nkd->nd', w, X[i]); n = f'knn{k}'
    hit = np.mean([[cat(tr[j]) == cat(te[r]) for j in i[r].tolist()] for r in range(len(te))])
    out[n] = {'nn_cat': float(hit)}; diag(n, Q); save(n, Q)
S, U = torch.linalg.eigh(X @ X.T)                                                  # Gram [1300, 1300], 오름차순
S, U = S.flip(0).clamp_min(1e-6), U.flip(1)
XP = X @ P.T                                                                       # [1300, 104]
for pc in map(int, a.pca.split(',')):
    Up = U[:, :pc]; Q = ((X.T @ Up) @ ((Up.T @ XP) / S[:pc, None])).T              # 주성분 pc개 공간 투영
    diag(f'pca{pc}', Q); save(f'pca{pc}', Q)
out['explained'] = {str(pc): float(S[:pc].sum() / S.sum()) for pc in map(int, a.pca.split(','))}
print('주성분 설명 비율', out['explained'], flush=True)
json.dump(out, open(f'{a.out}/tokfix_stats.json', 'w'), indent=1); print('TOKFIX_DONE', flush=True)
