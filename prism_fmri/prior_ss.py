# B안 3단계 (11-10, 5090): TRELLIS 형상 잠재(SS 잠재 z [8, 16³]) 위 조건 없는 prior (flow matching)
#  학습 데이터: ShapeNet 렌더 토큰 → TRELLIS 1단계 flow (seed 42) → z  (trellis_tools.py ss)
#  모델: 2³ 패치 → 512토큰 × 64 → DiT (adaLN) → 같은 모양 속도
#  train : <ss>/z/<uid>.npy (train_objs) → <out>/prior_final.pt
#  sample: N개 → <out>/gen/<gen_XXXX>.npy (z, fp16) → 점유는 trellis_tools.py ssdec, 검사는 occ_check.py
import os, math, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['train', 'sample'])
p.add_argument('--ss', required=True); p.add_argument('--objs', default=''); p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, default=20000); p.add_argument('--bs', type=int, default=64); p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--width', type=int, default=512); p.add_argument('--depth', type=int, default=12); p.add_argument('--heads', type=int, default=8)
p.add_argument('--dropout', type=float, default=0.0)
p.add_argument('--ckpt', default=''); p.add_argument('--n', type=int, default=64); p.add_argument('--n_steps', type=int, default=50)
p.add_argument('--temp', type=float, default=1.0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--cond', type=int, default=0)            # train: 1이면 범주 조건 (11-11, uid 앞 synset → 임베딩, 시간 임베딩에 더함)
p.add_argument('--p_uncond', type=float, default=0.1)    # train: 범주를 '없음'으로 바꾸는 비율 (CFG용)
p.add_argument('--cfg', type=float, default=1.0)         # sample: v = v_없음 + cfg · (v_범주 − v_없음), 1이면 조건만
a = p.parse_args()
dev = 'cuda'; C, R, PS = 8, 16, 2; L = (R // PS) ** 3; CP = C * PS ** 3


def patch(z):                                          # [B, 8, 16, 16, 16] → [B, 512, 64]
    B = len(z); z = z.reshape(B, C, R // PS, PS, R // PS, PS, R // PS, PS)
    return z.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, L, CP)


def unpatch(x):
    B = len(x); n = R // PS; x = x.reshape(B, n, n, n, C, PS, PS, PS)
    return x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, C, R, R, R)


def tfeat(t, h=256):
    f = torch.exp(-math.log(10000) * torch.arange(h // 2, device=t.device) / (h // 2)); x = t[:, None] * 1000 * f[None]
    return torch.cat([x.sin(), x.cos()], -1)


class Blk(nn.Module):
    def __init__(s, D, H, dp):
        super().__init__()
        s.n1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6); s.attn = nn.MultiheadAttention(D, H, batch_first=True, dropout=dp)
        s.n2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6); s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(approximate='tanh'), nn.Dropout(dp), nn.Linear(4 * D, D))
        s.ada = nn.Linear(D, 6 * D); nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)

    def forward(s, x, c):
        a1, b1, g1, a2, b2, g2 = s.ada(F.silu(c))[:, None].chunk(6, -1)
        h = s.n1(x) * (1 + b1) + a1; x = x + g1 * s.attn(h, h, h, need_weights=False)[0]
        h = s.n2(x) * (1 + b2) + a2; return x + g2 * s.mlp(h)


class DiT(nn.Module):
    def __init__(s, D, depth, H, dp, ncls=0):
        super().__init__()
        s.cls = nn.Embedding(ncls + 1, D) if ncls else None   # 마지막 번호 = 범주 없음
        s.inp = nn.Linear(CP, D); s.pos = nn.Parameter(torch.randn(1, L, D) * 0.02)
        s.t = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D)); s.blocks = nn.ModuleList([Blk(D, H, dp) for _ in range(depth)])
        s.no = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6); s.ada = nn.Linear(D, 2 * D); s.outp = nn.Linear(D, CP)
        for q in (s.ada, s.outp): nn.init.zeros_(q.weight); nn.init.zeros_(q.bias)

    def forward(s, z, t, y=None):
        h = s.inp(patch(z)) + s.pos; e = s.t(tfeat(t))
        if s.cls is not None: e = e + s.cls(y)
        for b in s.blocks: h = b(h, e)
        sh, sc = s.ada(F.silu(e))[:, None].chunk(2, -1); return unpatch(s.outp(s.no(h) * (1 + sc) + sh))


if a.mode == 'train':
    os.makedirs(a.out, exist_ok=True)
    objs = [l.strip() for l in open(a.objs) if l.strip() and os.path.exists(f'{a.ss}/z/{l.strip()}.npy')]
    Z = torch.from_numpy(np.stack([np.load(f'{a.ss}/z/{u}.npy') for u in objs])).float().to(dev)
    rng = np.random.RandomState(0); vm = np.zeros(len(Z), bool); vm[rng.choice(len(Z), 300, replace=False)] = True
    Zt, Zv = Z[torch.from_numpy(~vm).to(dev)], Z[torch.from_numpy(vm).to(dev)]
    CLASSES = sorted({u.split('_')[0] for u in objs}) if a.cond else []; NC = len(CLASSES)
    Y = torch.tensor([CLASSES.index(u.split('_')[0]) if a.cond else 0 for u in objs], device=dev)
    Yt, Yv = Y[torch.from_numpy(~vm).to(dev)], Y[torch.from_numpy(vm).to(dev)]
    MU, SD = Zt.mean((0, 2, 3, 4), keepdim=True), Zt.std((0, 2, 3, 4), keepdim=True) + 1e-4
    norm = lambda z: (z - MU) / SD
    net = DiT(a.width, a.depth, a.heads, a.dropout, NC).to(dev)
    print(f'[model] {sum(q.numel() for q in net.parameters()) / 1e6:.1f}M | train {len(Zt)} val {len(Zv)} | 범주 조건 {NC} | z 채널 표준편차 {SD.flatten().cpu().numpy().round(3)}', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))

    def loss_of(x0, y):
        x0 = norm(x0); t = torch.sigmoid(torch.randn(len(x0), device=dev)); tt = t.view(-1, 1, 1, 1, 1)
        e = torch.randn_like(x0); xt = (1 - tt) * x0 + tt * e
        with torch.autocast(dev, dtype=torch.bfloat16): v = net(xt, t, y)
        return F.mse_loss(v.float(), e - x0)
    t0 = time.time(); run = []
    for step in range(1, a.steps + 1):
        ix = torch.randint(len(Zt), (a.bs,), device=dev); x0, y = Zt[ix], Yt[ix].clone()
        if NC: y[torch.rand(len(y), device=dev) < a.p_uncond] = NC
        L_ = loss_of(x0, y); opt.zero_grad(set_to_none=True); L_.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); run.append(L_.item())
        if step % 500 == 0: print(f'[step {step}/{a.steps}] loss {np.mean(run):.4f} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB', flush=True); run = []
        if step % 2000 == 0 or step == a.steps:
            net.eval(); torch.manual_seed(0)
            with torch.no_grad(): vv = np.mean([loss_of(Zv[b:b + 100], Yv[b:b + 100]).item() for b in range(0, len(Zv), 100)])
            net.train(); print(f'[val] step {step} loss {vv:.4f}', flush=True)
            if step % 10000 == 0 and step < a.steps: torch.save({'net': net.state_dict(), 'args': vars(a), 'MU': MU.cpu(), 'SD': SD.cpu(), 'classes': CLASSES}, f'{a.out}/prior_step{step}.pt')
    torch.save({'net': net.state_dict(), 'args': vars(a), 'MU': MU.cpu(), 'SD': SD.cpu(), 'classes': CLASSES}, f'{a.out}/prior_final.pt'); print('PRIOR_TRAIN_DONE', flush=True)

else:
    import json
    ck = torch.load(a.ckpt, map_location='cpu'); ar = ck['args']; CLASSES = ck.get('classes', []); NC = len(CLASSES)
    net = DiT(ar['width'], ar['depth'], ar['heads'], 0.0, NC).to(dev).eval(); net.load_state_dict(ck['net'])
    MU, SD = ck['MU'].to(dev), ck['SD'].to(dev); os.makedirs(a.out, exist_ok=True)
    g = torch.Generator(device=dev).manual_seed(a.seed); ts = torch.linspace(1, 0, a.n_steps + 1, device=dev)
    yall = torch.arange(a.n, device=dev) % max(NC, 1); gcls = {}   # 범주 조건이면 gen_k의 범주 = k mod 범주 수
    for b in range(0, a.n, 32):
        B = min(32, a.n - b); x = a.temp * torch.randn(B, C, R, R, R, device=dev, generator=g); y = yall[b:b + B]

        def vel(x, t):
            if not NC: return net(x, t).float()
            vc = net(x, t, y).float()
            if a.cfg == 1: return vc
            vu = net(x, t, torch.full_like(y, NC)).float()
            return vu + a.cfg * (vc - vu)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            for t, tp in zip(ts[:-1], ts[1:]): x = x - (t - tp) * vel(x, t.expand(B))
        x = x.float() * SD + MU
        for j in range(B):
            np.save(f'{a.out}/gen_{b + j:04d}.npy', x[j].half().cpu().numpy())
            if NC: gcls[f'gen_{b + j:04d}'] = CLASSES[int(y[j])]
    if NC: json.dump(gcls, open(f'{a.out}/cls.json', 'w'), indent=1)
    print(f'생성 {a.n}개 → {a.out} (T {a.temp}, {a.n_steps} steps, 범주 조건 {NC}, cfg {a.cfg})', flush=True); print('PRIOR_SAMPLE_DONE', flush=True)
