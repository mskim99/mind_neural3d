# B안 2단계 (11-10, 5090): DINO 조건 토큰 [1374, 1024]을 작은 공간으로 압축 → prior는 그 공간에서
#  압축 = 채널 PCA (1024 → d) + 패치 37×37 평균 풀링 (s=1 그대로 / s=2 → 19×19)  [+ 선택: 물체당 벡터 전역 PCA → k]
#  복원 = PCA 역변환 (+ 쌍선형 업샘플) → 토큰별 layer_norm (TRELLIS get_cond와 같은 정규화)
#  fit   : 학습 토큰 n_fit개로 채널 PCA → <out>/chan_pca.pt (+ 설명 분산)
#  gfit  : --cfg c{d}_s{s}로 학습 토큰 전부 인코딩 → 전역 PCA (Gram 행렬) → <out>/glob_c{d}_s{s}.pt
#  recon : 검증 물체를 cfg별로 압축 → 복원 → <out>/recon/<cfg>/<uid>.npy + 지표 (토큰 cos, 복사 검사 거리의 cos)
#   cfg 이름: c64_s2 (채널 64, 풀링 2) / c64_s2_g1024 (그 위에 전역 PCA 1024)
import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['fit', 'gfit', 'recon'])
p.add_argument('--tok', required=True); p.add_argument('--objs', required=True); p.add_argument('--out', required=True)
p.add_argument('--n_fit', type=int, default=2000); p.add_argument('--cfg', default='c64_s2'); p.add_argument('--kmax', type=int, default=2048)
p.add_argument('--cfgs', default=''); p.add_argument('--n', type=int, default=16)
a = p.parse_args()
dev = 'cuda'; NP, G = 5, 37
ld = lambda u: torch.from_numpy(np.load(f'{a.tok}/{u}.npy').astype(np.float32))
OBJS = [l.strip() for l in open(a.objs) if l.strip()]


def parse(cfg):
    q = dict((t[0], int(t[1:])) for t in cfg.split('_')); return q['c'], q['s'], q.get('g', 0)


def load_pca():
    return torch.load(f'{a.out}/chan_pca.pt')


def enc(x, P, d, s):                                    # x [B, 1374, 1024] → 앞 [B, 5, d], 격자 [B, d, g, g]
    z = (x - P['mu']) @ P['V'][:, :d]
    pre, g = z[:, :NP], z[:, NP:].reshape(len(z), G, G, d).permute(0, 3, 1, 2)
    if s > 1: g = F.adaptive_avg_pool2d(g, (G + s - 1) // s)
    return pre, g


def dec(pre, g, P, d):                                  # → [B, 1374, 1024], 토큰별 LN
    if g.shape[-1] != G: g = F.interpolate(g, (G, G), mode='bilinear', align_corners=False)
    z = torch.cat([pre, g.flatten(2).transpose(1, 2)], 1)
    return F.layer_norm(z @ P['V'][:, :d].T + P['mu'], (1024,))


def flat(pre, g): return torch.cat([pre.flatten(1), g.flatten(1)], 1)


def unflat(v, d, gs):
    B = len(v); return v[:, :NP * d].reshape(B, NP, d), v[:, NP * d:].reshape(B, d, gs, gs)


if a.mode == 'fit':
    os.makedirs(a.out, exist_ok=True); rng = np.random.RandomState(0)
    pick = rng.choice(len(OBJS), min(a.n_fit, len(OBJS)), replace=False)
    S, S2, n = torch.zeros(1024, device=dev, dtype=torch.float64), torch.zeros(1024, 1024, device=dev, dtype=torch.float64), 0
    for i in pick:
        x = ld(OBJS[i]).to(dev).double(); S += x.sum(0); S2 += x.T @ x; n += len(x)
    mu = S / n; C = S2 / n - mu[:, None] * mu[None]
    ev, V = torch.linalg.eigh(C); ev, V = ev.flip(0), V.flip(1)
    torch.save({'mu': mu.float().cpu(), 'V': V.float().cpu(), 'ev': ev.float().cpu(), 'n_obj': len(pick)}, f'{a.out}/chan_pca.pt')
    cum = (ev.cumsum(0) / ev.sum()).cpu().numpy()
    print('채널 PCA 설명 분산:', ' · '.join(f'{d}: {cum[d - 1]:.3f}' for d in (8, 16, 32, 64, 128, 256, 512)), flush=True)
    print('FIT_DONE')

elif a.mode == 'gfit':
    P = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in load_pca().items()}; d, s, _ = parse(a.cfg)
    Z = []
    for b in range(0, len(OBJS), 64):
        x = torch.stack([ld(u) for u in OBJS[b:b + 64]]).to(dev)
        with torch.no_grad(): Z.append(flat(*enc(x, P, d, s)).cpu())
        if b % 1280 == 0: print(f'  인코딩 {b}/{len(OBJS)}', flush=True)
    Z = torch.cat(Z); m = Z.mean(0); Zc = (Z - m).to(dev).double()
    Gm = Zc @ Zc.T; ev, U = torch.linalg.eigh(Gm); ev, U = ev.flip(0).clamp_min(0), U.flip(1)
    k = min(a.kmax, len(Z) - 1)
    W = (Zc.T @ U[:, :k]) / ev[:k].sqrt()              # [D, k] 정규직교 주성분
    cum = (ev.cumsum(0) / ev.sum()).cpu().numpy()
    codes = (Zc @ W).float().cpu()                      # 학습 물체 코드 (prior 학습용)
    torch.save({'m': m, 'W': W.float().cpu(), 'ev': ev[:k].float().cpu(), 'cfg': a.cfg, 'codes': codes, 'objs': OBJS}, f'{a.out}/glob_{a.cfg}.pt')
    print(f'전역 PCA {a.cfg}: 차원 {Z.shape[1]}, 물체 {len(Z)} | 설명 분산', ' · '.join(f'{q}: {cum[q - 1]:.3f}' for q in (64, 256, 512, 1024, 2048) if q <= k), flush=True)
    print('GFIT_DONE')

else:
    P = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in load_pca().items()}
    objs = OBJS[:a.n]; X = torch.stack([ld(u) for u in objs]).to(dev); R = {}

    def rep(t):
        t = (t - t.mean(-1, keepdim=True)) / torch.sqrt(t.var(-1, keepdim=True, unbiased=False) + 1e-5); return t.mean(1)
    for cfg in a.cfgs.split(','):
        d, s, k = parse(cfg); od = f'{a.out}/recon/{cfg}'; os.makedirs(od, exist_ok=True)
        with torch.no_grad():
            pre, g = enc(X, P, d, s)
            if k:
                GB = torch.load(f'{a.out}/glob_c{d}_s{s}.pt'); W = GB['W'][:, :k].to(dev); m = GB['m'].to(dev)
                v = ((flat(pre, g) - m) @ W) @ W.T + m; pre, g = unflat(v, d, g.shape[-1])
            Y = dec(pre, g, P, d)
        tc = float(F.cosine_similarity(Y, X, dim=-1).mean()); rc = float(F.cosine_similarity(rep(Y), rep(X), dim=-1).mean())
        ndim = (NP + g.shape[-1] ** 2) * d if not k else k
        R[cfg] = {'token_cos': round(tc, 4), 'rep_cos': round(rc, 4), 'latent_dim': ndim}
        for j, u in enumerate(objs): np.save(f'{od}/{u}.npy', Y[j].half().cpu().numpy())
        print(f'[recon] {cfg:16s} 잠재 차원 {ndim:7d} | 토큰 cos {tc:.3f} | 평균 토큰 cos {rc:.3f}', flush=True)
    json.dump(R, open(f'{a.out}/recon/metrics.json', 'w'), indent=1)
    open(f'{a.out}/recon/names.txt', 'w').write('\n'.join(objs) + '\n')
    print('RECON_DONE')
