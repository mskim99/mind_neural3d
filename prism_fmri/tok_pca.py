# prior 0단계 (b) (11-09, GPU): 정답 토큰을 PCA-k로 압축했다가 복원해도 TRELLIS가 색을 내는지 → 압축 공간 prior(PCA 공간 flow)가 가능한지 판단
#  PCA = train 1300개 정답 토큰(프레임 0, [1374, 1024] 펼침) — Gram 행렬(N×N) 고유분해, 평균 빼고. 복원 x̂ = μ + U_k U_kᵀ (x − μ)
#  test 정답 토큰마다 k별 복원 → <out>/pca_k<k>/<uid>.npy (fp16). 설명 분산 비율과 test 복원 상대 오차를 출력
# 사용: python tok_pca.py --tgt <shape_targets> --data_root <fMRI-Shape> --out <폴더> [--ks 64,256,1024] [--names <uid 목록>]
import os, json, argparse, numpy as np, torch
p = argparse.ArgumentParser()
p.add_argument('--tgt', required=True); p.add_argument('--data_root', required=True); p.add_argument('--out', required=True)
p.add_argument('--ks', default='64,256,1024'); p.add_argument('--names', default='')
a = p.parse_args()
ks = [int(k) for k in a.ks.split(',')]
lst = lambda sp: [l.strip().split('/')[1] for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
tr = [u for u in lst('train') if os.path.exists(f'{a.tgt}/dino/{u}.npy')]
te = [l.split()[-1] for l in open(a.names) if l.strip()] if a.names else lst('test')
te = [u for u in te if os.path.exists(f'{a.tgt}/dino/{u}.npy')]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.stack([torch.from_numpy(np.load(f'{a.tgt}/dino/{u}.npy')).reshape(-1) for u in tr]).to(dev).float()   # [N, D]
shape = np.load(f'{a.tgt}/dino/{tr[0]}.npy').shape
mu = X.mean(0); X -= mu
G = X @ X.t()                                                          # [N, N]
ev, V = torch.linalg.eigh(G.double()); ev, V = ev.flip(0).clamp_min(0), V.flip(1)
kmax = min(max(ks), len(tr) - 1)
U = (X.t() @ V[:, :kmax].float()) / ev[:kmax].sqrt().float()[None]  # [D, kmax], 정규직교 주성분
del X, G
tot = float(ev.sum()); R = {'n_train': len(tr), 'shape': list(shape), 'explained': {}, 'test_rel_err': {}}
for k in ks:
    k2 = min(k, kmax); R['explained'][k] = float(ev[:k2].sum()) / tot; os.makedirs(f'{a.out}/pca_k{k}', exist_ok=True); errs = []
    for u in te:
        x = torch.from_numpy(np.load(f'{a.tgt}/dino/{u}.npy')).reshape(-1).to(dev).float() - mu
        r = U[:, :k2] @ (U[:, :k2].t() @ x)
        errs.append(float((x - r).norm() / x.norm()))
        np.save(f'{a.out}/pca_k{k}/{u}.npy', (r + mu).reshape(shape).half().cpu().numpy())
    R['test_rel_err'][k] = float(np.mean(errs))
    print(f'PCA k {k}: train 설명 분산 {R["explained"][k]:.3f} | test 복원 상대 오차 {R["test_rel_err"][k]:.3f} (n {len(te)})', flush=True)
json.dump(R, open(f'{a.out}/pca_stats.json', 'w', encoding='utf-8'), indent=1)
print('PCA_DONE', flush=True)
