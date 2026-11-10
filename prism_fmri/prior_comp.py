# B안 2단계 (11-10, 5090): 압축 공간 조건 없는 prior (flow matching) → 복원 → DINO 토큰 → 복사 검사
#  공간 (tok_compress.py와 같은 압축)
#   glob : 전역 PCA 코드 k차원 (주성분 표준편차로 나눠 백색화) → 잔차 MLP
#   grid : 채널 PCA d × 풀링 s 격자 [5 + g², d] (채널별 표준화) → 작은 DiT
#  train  : <comp>/chan_pca.pt (+ glob_<cfg>.pt) 사용, 학습 물체 = prior_u/train_objs.txt
#  sample : N개 생성 → 복원 · LN → <out>/gen_XXXX.npy + 복사 검사 (prior_u와 같은 은행 · 거리)
#   --gauss 1 (glob만): 학습 없이 주성분 가우시안에서 뽑기 (기준선)
import os, sys, json, math, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['train', 'sample'])
p.add_argument('--space', choices=['glob', 'grid'], default='glob'); p.add_argument('--cfg', default='c64_s2'); p.add_argument('--k', type=int, default=512)
p.add_argument('--comp', required=True); p.add_argument('--tok', required=True); p.add_argument('--bankdir', required=True)   # prior_u (train/val 목록 · copy_bank)
p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, default=30000); p.add_argument('--bs', type=int, default=64); p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--width', type=int, default=2048); p.add_argument('--depth', type=int, default=8); p.add_argument('--heads', type=int, default=8)
p.add_argument('--dropout', type=float, default=0.1); p.add_argument('--white', type=int, default=1)   # glob: 1 = 주성분별 백색화, 0 = 한 배율 (분산 비율 유지, 꼬리 잡음 안 키움)
p.add_argument('--n', type=int, default=64); p.add_argument('--n_steps', type=int, default=50); p.add_argument('--temp', type=float, default=1.0)
p.add_argument('--ckpt', default=''); p.add_argument('--gauss', type=int, default=0); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
dev = 'cuda'; NP, G = 5, 37
ld = lambda u: torch.from_numpy(np.load(f'{a.tok}/{u}.npy').astype(np.float32))
d_ch, s_pool = [int(t[1:]) for t in a.cfg.split('_')[:2]]; gs = (G + s_pool - 1) // s_pool
P = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in torch.load(f'{a.comp}/chan_pca.pt').items()}
V = P['V'][:, :d_ch]


def enc(x):
    z = (x - P['mu']) @ V; pre, g = z[:, :NP], z[:, NP:].reshape(len(z), G, G, d_ch).permute(0, 3, 1, 2)
    if s_pool > 1: g = F.adaptive_avg_pool2d(g, gs)
    return torch.cat([pre, g.flatten(2).transpose(1, 2)], 1)          # [B, 5 + g², d]


def dec(z):                                            # [B, 5 + g², d] → LN 토큰 [B, 1374, 1024]
    pre, g = z[:, :NP], z[:, NP:].transpose(1, 2).reshape(len(z), d_ch, gs, gs)
    if gs != G: g = F.interpolate(g, (G, G), mode='bilinear', align_corners=False)
    z = torch.cat([pre, g.flatten(2).transpose(1, 2)], 1)
    return F.layer_norm(z @ V.T + P['mu'], (1024,))


if a.space == 'glob':
    GB = torch.load(f'{a.comp}/glob_{a.cfg}.pt'); W = GB['W'][:, :a.k].to(dev); m = GB['m'].to(dev); sdv = GB['ev'][:a.k].to(dev).sqrt() / math.sqrt(len(GB['objs']))
    if not a.white: sdv = sdv.pow(2).mean().sqrt().expand_as(sdv).clone()   # 전체 분산 = k
    to_lat = lambda c: c / sdv                          # 코드 → 백색 잠재
    from_lat = lambda y: dec(_unflat((y * sdv) @ W.T + m))


def _unflat(v):                                        # tok_compress.flat의 역: [앞 5×d (토큰순), 격자 d×g² (채널순)]
    B = len(v); pre = v[:, :NP * d_ch].reshape(B, NP, d_ch); g = v[:, NP * d_ch:].reshape(B, d_ch, gs * gs).transpose(1, 2)
    return torch.cat([pre, g], 1)


def _flat(z):
    return torch.cat([z[:, :NP].flatten(1), z[:, NP:].transpose(1, 2).flatten(1)], 1)


def tfeat(t, h=256):
    f = torch.exp(-math.log(10000) * torch.arange(h // 2, device=t.device) / (h // 2)); x = t[:, None] * 1000 * f[None]
    return torch.cat([x.sin(), x.cos()], -1)


class ResMLP(nn.Module):                               # glob: [B, k]
    def __init__(s, k, W, depth, dp):
        super().__init__()
        s.inp = nn.Linear(k, W); s.t = nn.Sequential(nn.Linear(256, W), nn.SiLU(), nn.Linear(W, W))
        s.blocks = nn.ModuleList([nn.ModuleDict({'n': nn.LayerNorm(W, elementwise_affine=False), 'ada': nn.Linear(W, 3 * W),
                                                 'f': nn.Sequential(nn.Linear(W, 2 * W), nn.GELU(), nn.Dropout(dp), nn.Linear(2 * W, W))}) for _ in range(depth)])
        for b in s.blocks: nn.init.zeros_(b['ada'].weight); nn.init.zeros_(b['ada'].bias)
        s.no = nn.LayerNorm(W, elementwise_affine=False); s.outp = nn.Linear(W, k); nn.init.zeros_(s.outp.weight); nn.init.zeros_(s.outp.bias)

    def forward(s, x, t):
        h = s.inp(x); e = F.silu(s.t(tfeat(t)))
        for b in s.blocks:
            sh, sc, g = b['ada'](e).chunk(3, -1); h = h + g * b['f'](b['n'](h) * (1 + sc) + sh)
        return s.outp(s.no(h))


class Blk(nn.Module):
    def __init__(s, D, H):
        super().__init__()
        s.n1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6); s.attn = nn.MultiheadAttention(D, H, batch_first=True)
        s.n2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6); s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(approximate='tanh'), nn.Linear(4 * D, D))
        s.ada = nn.Linear(D, 6 * D); nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)

    def forward(s, x, c):
        a1, b1, g1, a2, b2, g2 = s.ada(F.silu(c))[:, None].chunk(6, -1)
        h = s.n1(x) * (1 + b1) + a1; x = x + g1 * s.attn(h, h, h, need_weights=False)[0]
        h = s.n2(x) * (1 + b2) + a2; return x + g2 * s.mlp(h)


class DiT(nn.Module):                                  # grid: [B, 5 + g², d]
    def __init__(s, C, L, D, depth, H):
        super().__init__()
        s.inp = nn.Linear(C, D); s.pos = nn.Parameter(torch.randn(1, L, D) * 0.02)
        s.t = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D)); s.blocks = nn.ModuleList([Blk(D, H) for _ in range(depth)])
        s.no = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6); s.ada = nn.Linear(D, 2 * D); s.outp = nn.Linear(D, C)
        for q in (s.ada, s.outp): nn.init.zeros_(q.weight); nn.init.zeros_(q.bias)

    def forward(s, x, t):
        h = s.inp(x) + s.pos; e = s.t(tfeat(t))
        for b in s.blocks: h = b(h, e)
        sh, sc = s.ada(F.silu(e))[:, None].chunk(2, -1); return s.outp(s.no(h) * (1 + sc) + sh)


def make_net():
    if a.space == 'glob': return ResMLP(a.k, a.width, a.depth, a.dropout).to(dev)
    return DiT(d_ch, NP + gs * gs, a.width, a.depth, a.heads).to(dev)


def latents(objs):                                     # 학습 잠재 (grid: 채널별 표준화 전 값, 한 번 만들어 저장)
    if a.space == 'glob':
        i = {u: j for j, u in enumerate(GB['objs'])}; return to_lat(GB['codes'][[i[u] for u in objs], :a.k].to(dev))
    f = f'{a.comp}/grid_{a.cfg}.pt'
    if os.path.exists(f): Z = torch.load(f)
    else:
        allo = [l.strip() for l in open(f'{a.bankdir}/train_objs.txt') if l.strip()]; Z = []
        with torch.no_grad():
            for b in range(0, len(allo), 64): Z.append(enc(torch.stack([ld(u) for u in allo[b:b + 64]]).to(dev)).half().cpu())
        Z = {'z': torch.cat(Z), 'objs': allo}; torch.save(Z, f)
    i = {u: j for j, u in enumerate(Z['objs'])}; return Z['z'][[i[u] for u in objs]].float().to(dev)


TR = [l.strip() for l in open(f'{a.bankdir}/train_objs.txt') if l.strip()]

if a.mode == 'train':
    os.makedirs(a.out, exist_ok=True); X = latents(TR)
    rng = np.random.RandomState(0); vi = torch.from_numpy(rng.choice(len(X), 300, replace=False)).to(dev)
    vm = torch.zeros(len(X), dtype=torch.bool, device=dev); vm[vi] = True; Xv, Xt = X[vm], X[~vm]   # 학습 안 쪽 검증 (손실만)
    if a.space == 'grid': MU, SD = Xt.mean((0, 1)), Xt.std((0, 1)) + 1e-4
    else: MU, SD = torch.zeros(1, device=dev), torch.ones(1, device=dev)
    norm = lambda x: (x - MU) / SD
    net = make_net(); print(f'[model] {a.space} {a.cfg} k={a.k} | {sum(q.numel() for q in net.parameters()) / 1e6:.1f}M | train {len(Xt)} val {len(Xv)} | 잠재 {tuple(X.shape[1:])}', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))

    def loss_of(x0):
        x0 = norm(x0); sh = (-1,) + (1,) * (x0.dim() - 1)
        t = torch.sigmoid(torch.randn(len(x0), device=dev)); e = torch.randn_like(x0); xt = (1 - t.view(sh)) * x0 + t.view(sh) * e
        with torch.autocast(dev, dtype=torch.bfloat16): v = net(xt, t)
        return F.mse_loss(v.float(), e - x0)
    t0 = time.time(); run = []
    for step in range(1, a.steps + 1):
        L = loss_of(Xt[torch.randint(len(Xt), (a.bs,), device=dev)])
        opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); run.append(L.item())
        if step % 500 == 0: print(f'[step {step}/{a.steps}] loss {np.mean(run):.4f} {time.time() - t0:.0f}s', flush=True); run = []
        if step % 2500 == 0 or step == a.steps:
            net.eval(); torch.manual_seed(0)
            with torch.no_grad(): vv = np.mean([loss_of(Xv[b:b + 100]).item() for b in range(0, len(Xv), 100)])
            net.train(); print(f'[val] step {step} loss {vv:.4f}', flush=True)
    torch.save({'net': net.state_dict(), 'args': vars(a), 'MU': MU.cpu(), 'SD': SD.cpu()}, f'{a.out}/prior_final.pt'); print('PRIOR_TRAIN_DONE', flush=True)

else:
    os.makedirs(a.out, exist_ok=True); g = torch.Generator(device=dev).manual_seed(a.seed); gen = []
    if a.gauss:
        y = a.temp * torch.randn(a.n, a.k, device=dev, generator=g) * (GB['ev'][:a.k].to(dev).sqrt() / math.sqrt(len(GB['objs'])) / sdv); tag = f'가우시안 (학습 없음) k={a.k}'
        with torch.no_grad(): gen = [x.cpu() for b in range(0, a.n, 16) for x in from_lat(y[b:b + 16])]
    else:
        ck = torch.load(a.ckpt, map_location='cpu')   # --space · --cfg · --k는 학습 때와 같게 넘김
        for kk in ('width', 'depth', 'heads'): setattr(a, kk, ck['args'][kk])
        assert ck['args'].get('white', 1) == a.white, '--white를 학습 때와 같게'
        net = make_net().eval(); net.load_state_dict(ck['net']); MU, SD = ck['MU'].to(dev), ck['SD'].to(dev)
        shape = (a.k,) if a.space == 'glob' else (NP + gs * gs, d_ch); tag = f'prior {a.space} {a.cfg} k={a.k} T={a.temp}'
        ts = torch.linspace(1, 0, a.n_steps + 1, device=dev)
        for b in range(0, a.n, 16):
            B = min(16, a.n - b); x = a.temp * torch.randn((B,) + shape, device=dev, generator=g)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                for t, tp in zip(ts[:-1], ts[1:]): x = x - (t - tp) * net(x, t.expand(B)).float()
            x = x.float() * SD + MU
            with torch.no_grad(): gen += [z.cpu() for z in (from_lat(x) if a.space == 'glob' else dec(x))]
    for j, x in enumerate(gen): np.save(f'{a.out}/gen_{j:04d}.npy', x.half().numpy())

    def rep(t):
        t = t.float(); t = (t - t.mean(-1, keepdim=True)) / torch.sqrt(t.var(-1, keepdim=True, unbiased=False) + 1e-5); return t.mean(-2)
    BK = torch.load(f'{a.bankdir}/copy_bank.pt'); TM = BK['rep'].mean(0); nz = lambda v: F.normalize(v - TM, dim=-1); TRn = nz(BK['rep']).to(dev); TM = TM.to(dev)
    gc = [float((TRn @ nz(rep(x.to(dev)))).max()) for x in gen]
    vo = [l.strip() for l in open(f'{a.bankdir}/val_objs.txt') if l.strip()][:200]
    with torch.no_grad():
        rr = [float((TRn @ nz(rep(ld(u).to(dev)))).max()) for u in vo]
        if a.space == 'glob': rc = [float((TRn @ nz(rep(from_lat(to_lat((_flat(enc(ld(u)[None].to(dev))) - m) @ W))[0]))).max()) for u in vo]
        else: rc = [float((TRn @ nz(rep(dec(enc(ld(u)[None].to(dev)))[0]))).max()) for u in vo]
    # 다양성: 생성끼리 평균 cos (같은 거리)
    Gn = torch.stack([nz(rep(x.to(dev))) for x in gen]); S = Gn @ Gn.T; div = float((S.sum() - S.diag().sum()) / (len(gen) * (len(gen) - 1)))
    Rn = torch.stack([nz(rep(ld(u).to(dev))) for u in vo[:64]]); S2 = Rn @ Rn.T; rdiv = float((S2.sum() - S2.diag().sum()) / (len(Rn) * (len(Rn) - 1)))
    R = {'tag': tag, 'mean_gen': float(np.mean(gc)), 'mean_real_val': float(np.mean(rr)), 'mean_recon_val': float(np.mean(rc)), 'gen_pair_cos': div, 'real_pair_cos': rdiv, 'gen': gc}
    json.dump(R, open(f'{a.out}/copy_check.json', 'w'), indent=1)
    print(f'COPY [{tag}] 생성 {np.mean(gc):.3f} (최소 {np.min(gc):.3f} 최대 {np.max(gc):.3f}) | 검증 실제 {np.mean(rr):.3f} · 검증 압축 복원 {np.mean(rc):.3f} | 생성끼리 cos {div:.3f} (실제끼리 {rdiv:.3f})', flush=True)
    print('PRIOR_SAMPLE_DONE', flush=True)
