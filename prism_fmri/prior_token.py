# 토큰 prior 1단계 (11-09, MinD-3D++ venv, GPU): PRISM 예측을 조건으로 실제 TRELLIS 조건 토큰 [1374, 1024]을 생성하는 조건부 flow (토큰prior_설계.md)
#  쓰는 곳: TRELLIS 외형 단계(SLat) 조건. 형상 단계(SS)는 PRISM 예측 토큰 그대로 (trellis_tools.py --slat_pred)
#  조건 (모델이 달라도 의미가 같은 것만): 예측 토큰 요약 tok_pool [54, 1024] (token_util.pool) + 범주 점수 cls_score [13] (CLIP 텍스트 프로토타입 cos)
#    학습 조건 = 2겹 교차 PRISM의 '안 본' train 예측 (train_prism.py / infer_prism.py --fold). 같은 물체를 본 모델의 예측은 정답을 외워서 쓰지 않음
#  정답: 그 물체 자극 영상 프레임 6장 중 하나의 실제 토큰 (<tgt>/dino = 프레임 0, <tgt>/dino_v1~5)
#  모델: 정답 토큰 → token_util.merge (앞 5 + 패치 2×2 묶음 361) → DiT (조건 55토큰을 앞에 붙임, 시간은 adaLN-Zero) → 속도 v. 채널별 표준화 (train 정답 통계)
#  flow: x_t = (1 − t)·x0 + t·ε, v = ε − x0, t ~ logit-normal. 조건은 확률 p_drop으로 비움 (CFG). 조건 노이즈 증강 --cond_noise
#  train : python prior_token.py train --tgt <shape_targets> --oof <comp.npz,...> --out <폴더> [--steps 60000 --bs 32]
#  sample: python prior_token.py sample --ckpt <폴더>/prior_final.pt --comp <components.npz> --out <tokens 폴더> [--cfg 2 --steps 50 --names <uid 목록>]
#          → <out>/<uid>.npy (fp16, 토큰별 layer_norm 다시 적용 --ln 1 = 실제 조건 토큰과 같은 정규화) + <out>/copy_check.json:
#            생성 토큰과 가장 가까운 '학습에 쓴' 정답 토큰(학습 물체 × 학습 뷰 전부, 검증 물체 제외)의 cos (slat_retrieve.py와 같은 거리)과, 그 물체 실제 토큰(프레임 0)의 같은 값 (복사 여부)
import os, sys, json, math, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_util import merge, unmerge, pool

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['train', 'sample'])
p.add_argument('--tgt', default=''); p.add_argument('--oof', default=''); p.add_argument('--out', required=True)
p.add_argument('--views', default='0,1,2,3,4,5'); p.add_argument('--steps', type=int, default=60000); p.add_argument('--bs', type=int, default=32)
p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--warmup', type=int, default=1000); p.add_argument('--workers', type=int, default=8)
p.add_argument('--dim', type=int, default=768); p.add_argument('--depth', type=int, default=12); p.add_argument('--heads', type=int, default=12)
p.add_argument('--p_drop', type=float, default=0.2); p.add_argument('--cond_noise', type=float, default=0.1)
p.add_argument('--val_frac', type=float, default=0.05); p.add_argument('--val_every', type=int, default=2000); p.add_argument('--save_every', type=int, default=5000)
p.add_argument('--resume', action='store_true')
p.add_argument('--ckpt', default=''); p.add_argument('--comp', default=''); p.add_argument('--names', default='')
p.add_argument('--cfg', type=float, default=2.0); p.add_argument('--n_steps', type=int, default=50); p.add_argument('--seed', type=int, default=0)
p.add_argument('--data_root', default='')   # sample: copy_check용 train 목록 (비우면 copy_check 안 함)
p.add_argument('--ln', type=int, default=1)    # sample: 생성 토큰에 토큰별 affine 없는 layer_norm (TRELLIS get_cond 출력과 같은 정규화)
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'   # cpu는 로컬 점검용
NC = 13


def tfeat(d, h):                                       # 시간 임베딩 (sinusoidal)
    f = torch.exp(-math.log(10000) * torch.arange(h // 2, device=d.device) / (h // 2))
    x = d[:, None] * 1000 * f[None]
    return torch.cat([x.sin(), x.cos()], -1)


class Block(nn.Module):
    def __init__(s, D, H):
        super().__init__()
        s.n1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6); s.attn = nn.MultiheadAttention(D, H, batch_first=True)
        s.n2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6); s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(approximate='tanh'), nn.Linear(4 * D, D))
        s.ada = nn.Linear(D, 6 * D); nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)

    def forward(s, x, c):
        a1, b1, g1, a2, b2, g2 = s.ada(F.silu(c))[:, None].chunk(6, -1)
        h = s.n1(x) * (1 + b1) + a1; x = x + g1 * s.attn(h, h, h, need_weights=False)[0]
        h = s.n2(x) * (1 + b2) + a2; return x + g2 * s.mlp(h)


class PriorNet(nn.Module):
    def __init__(s, D=768, depth=12, H=12, C=1024):
        super().__init__()
        s.D = D
        s.in_pre, s.in_g = nn.Linear(C, D), nn.Linear(4 * C, D)
        s.pos_x = nn.Parameter(torch.randn(1, 5 + 361, D) * 0.02)
        s.in_c, s.in_s = nn.Linear(C, D), nn.Linear(NC, D)
        s.pos_c = nn.Parameter(torch.randn(1, 54 + 1, D) * 0.02)
        s.null = nn.Parameter(torch.zeros(1, 54 + 1, D))   # 조건 없음 (CFG)
        s.t_mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        s.blocks = nn.ModuleList([Block(D, H) for _ in range(depth)])
        s.n_out = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6); s.ada_out = nn.Linear(D, 2 * D)
        s.out_pre, s.out_g = nn.Linear(D, C), nn.Linear(D, 4 * C)
        for m in (s.ada_out, s.out_pre, s.out_g): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(s, xt, t, ctok, cscore, drop):          # xt [B, 1374, C] (표준화), t [B], ctok [B, 54, C], cscore [B, 13], drop [B] bool
        pre, g = merge(xt)
        x = torch.cat([s.in_pre(pre), s.in_g(g)], 1) + s.pos_x
        c = torch.cat([s.in_c(ctok), s.in_s(cscore)[:, None]], 1) + s.pos_c
        c = torch.where(drop[:, None, None], s.null.expand(len(c), -1, -1), c)
        e = s.t_mlp(tfeat(t, 256))
        h = torch.cat([c, x], 1); nc = c.shape[1]
        for b in s.blocks: h = b(h, e)
        sh, sc = s.ada_out(F.silu(e))[:, None].chunk(2, -1)
        h = s.n_out(h[:, nc:]) * (1 + sc) + sh
        return unmerge(s.out_pre(h[:, :5]), s.out_g(h[:, 5:]))


def ld(f): return torch.from_numpy(np.load(f).astype(np.float32))


if a.mode == 'train':
    os.makedirs(a.out, exist_ok=True)
    views = [int(v) for v in a.views.split(',')]
    vdir = lambda v: f'{a.tgt}/dino' if v == 0 else f'{a.tgt}/dino_v{v}'
    C_TOK, C_SC, NAMES = [], [], []
    for f in a.oof.split(','):                         # 피험자 · fold별 components.npz (안 본 train 물체)
        z = np.load(f); C_TOK.append(z['tok_pool']); C_SC.append(z['cls_score']); NAMES += [k.split('/')[1] for k in z['name']]
    C_TOK = np.concatenate(C_TOK); C_SC = np.concatenate(C_SC).astype(np.float32)
    ok = [i for i, u in enumerate(NAMES) if all(os.path.exists(f'{vdir(v)}/{u}.npy') for v in views)]
    objs = sorted({NAMES[i] for i in ok}); rng = np.random.RandomState(0)
    val_objs = set(rng.choice(objs, max(1, int(len(objs) * a.val_frac)), replace=False))
    tr = [i for i in ok if NAMES[i] not in val_objs]; va = [i for i in ok if NAMES[i] in val_objs]
    NL = chr(10)
    open(f'{a.out}/val_objs.txt', 'w').write(NL.join(sorted(val_objs)) + NL)          # 학습에서 뺀 물체 (cfg 조정에 씀)
    open(f'{a.out}/train_objs.txt', 'w').write(NL.join(sorted({NAMES[i] for i in tr})) + NL)   # 복사 검사 기준
    print(f'[data] 조건 {len(NAMES)}개 (정답 다 있는 것 {len(ok)}) | 물체 {len(objs)} (검증 {len(val_objs)}) | 뷰 {views} | train {len(tr)} · val {len(va)}', flush=True)
    st = torch.stack([ld(f'{vdir(v)}/{u}.npy') for u in list(objs)[:200] for v in views[:2]])          # 채널별 표준화 통계 (정답 토큰)
    MU, SD = st.mean((0, 1)), st.std((0, 1)) + 1e-4; del st
    torch.save({'mu': MU, 'sd': SD}, f'{a.out}/norm.pt')

    class DS(torch.utils.data.Dataset):
        def __init__(s, idx): s.idx = idx
        def __len__(s): return len(s.idx)
        def __getitem__(s, j):
            i = s.idx[j]; v = views[np.random.randint(len(views))]
            return ld(f'{vdir(v)}/{NAMES[i]}.npy'), torch.from_numpy(C_TOK[i].astype(np.float32)), torch.from_numpy(C_SC[i])

    tl = torch.utils.data.DataLoader(DS(tr), batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers, persistent_workers=a.workers > 0)
    vl = torch.utils.data.DataLoader(DS(va), batch_size=a.bs, shuffle=False, num_workers=min(2, a.workers))
    net = PriorNet(a.dim, a.depth, a.heads).to(dev)
    print(f'[model] {sum(q.numel() for q in net.parameters()) / 1e6:.1f}M', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    step = 0; latest = f'{a.out}/prior_latest.pt'
    if a.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location='cpu'); net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); step = ck['step']
        print('resumed', step, flush=True)
    MUd, SDd = MU.to(dev), SD.to(dev)
    norm = lambda x: (x - MUd) / SDd

    def loss_of(x0, ct, cs, train):
        B = len(x0); x0 = norm(x0); ct = norm(ct)
        if train and a.cond_noise > 0:
            ct = ct + a.cond_noise * torch.randn_like(ct); cs = cs + a.cond_noise * cs.std() * torch.randn_like(cs)
        t = torch.sigmoid(torch.randn(B, device=dev))  # logit-normal
        e = torch.randn_like(x0); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * e
        drop = (torch.rand(B, device=dev) < a.p_drop) if train else torch.zeros(B, dtype=torch.bool, device=dev)
        with torch.autocast(dev, dtype=torch.bfloat16):
            v = net(xt, t, ct, cs, drop)
        return F.mse_loss(v.float(), e - x0)

    log = open(f'{a.out}/train_log.csv', 'a'); t0 = time.time(); run = []
    while step < a.steps:
        for x0, ct, cs in tl:
            x0, ct, cs = x0.to(dev, non_blocking=True), ct.to(dev, non_blocking=True), cs.to(dev, non_blocking=True)
            L = loss_of(x0, ct, cs, True)
            opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
            run.append(L.item())
            if step % 50 == 0:
                print(f'[step {step}/{a.steps}] loss {np.mean(run):.4f} lr {sched.get_last_lr()[0]:.2e} mem {torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0:.1f}GB {time.time() - t0:.0f}s', flush=True)
                log.write(f'{step},train,{np.mean(run):.5f}\n'); log.flush(); run = []
            if step % a.val_every == 0 or step == a.steps:
                net.eval(); torch.manual_seed(0)
                with torch.no_grad(): vv = np.mean([loss_of(x.to(dev), c.to(dev), s.to(dev), False).item() for x, c, s in vl])
                net.train(); torch.manual_seed(step)
                print(f'[val] step {step} loss {vv:.4f} (검증 물체 {len(val_objs)}개, 조건 있음)', flush=True); log.write(f'{step},val,{vv:.5f}\n'); log.flush()
            if step % a.save_every == 0 or step == a.steps:
                torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step}, latest)
            if step >= a.steps: break
    torch.save({'net': net.state_dict(), 'args': vars(a), 'mu': MU, 'sd': SD}, f'{a.out}/prior_final.pt')
    print('PRIOR_TRAIN_DONE', flush=True)

else:
    ck = torch.load(a.ckpt, map_location='cpu'); ar = ck['args']
    net = PriorNet(ar['dim'], ar['depth'], ar['heads']).to(dev).eval(); net.load_state_dict(ck['net'])
    MU, SD = ck['mu'].to(dev), ck['sd'].to(dev)
    z = np.load(a.comp); names = [k.split('/')[1] for k in z['name']]
    want = [l.split()[-1] for l in open(a.names) if l.strip()] if a.names else names
    idx = [names.index(u) for u in want if u in names]
    os.makedirs(a.out, exist_ok=True); g = torch.Generator(device=dev).manual_seed(a.seed); gen = {}; sd_tok = []
    for b in range(0, len(idx), 16):
        ii = idx[b:b + 16]; B = len(ii)
        ct = (torch.from_numpy(z['tok_pool'][ii].astype(np.float32)).to(dev) - MU) / SD
        cs = torch.from_numpy(z['cls_score'][ii].astype(np.float32)).to(dev)
        x = torch.randn(B, 1374, 1024, device=dev, generator=g)
        ts = torch.linspace(1, 0, a.n_steps + 1, device=dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            for t, tp in zip(ts[:-1], ts[1:]):
                tt = t.expand(B)
                vc = net(x, tt, ct, cs, torch.zeros(B, dtype=torch.bool, device=dev)).float()
                vu = net(x, tt, ct, cs, torch.ones(B, dtype=torch.bool, device=dev)).float() if a.cfg != 1 else vc
                x = x - (t - tp) * (vu + a.cfg * (vc - vu))
        x = x.float() * SD + MU
        sd_tok.append(float(x.std(-1).mean()))
        if a.ln: x = F.layer_norm(x, (x.shape[-1],))
        for j, i in enumerate(ii):
            np.save(f'{a.out}/{names[i]}.npy', x[j].half().cpu().numpy()); gen[names[i]] = x[j].cpu()
    print(f'생성 {len(gen)}개 → {a.out} (cfg {a.cfg}, {a.n_steps} steps, LN 전 토큰별 표준편차 평균 {np.mean(sd_tok):.3f} · 실제 토큰은 1, LN {a.ln})', flush=True)
    if a.data_root and a.tgt:                          # 복사 검사: slat_retrieve.py와 같은 거리 (LN · 토큰 평균 · train 평균 빼기 · cos)
        def rep(t):
            t = t.float(); t = (t - t.mean(-1, keepdim=True)) / torch.sqrt(t.var(-1, keepdim=True, unbiased=False) + 1e-5); return t.mean(0)
        cdir = os.path.dirname(os.path.abspath(a.ckpt)); bank_f = f'{cdir}/copy_bank.pt'
        if os.path.exists(bank_f): BK = torch.load(bank_f)
        else:                                          # 학습에 쓴 정답 토큰 전부 (학습 물체 × 학습 뷰). 처음 한 번 만들어 저장
            views = [int(v) for v in str(ar['views']).split(',')]
            vdir = lambda v: f'{a.tgt}/dino' if v == 0 else f'{a.tgt}/dino_v{v}'
            objs = [l.strip() for l in open(f'{cdir}/train_objs.txt') if l.strip()]
            rows = [(u, v) for u in objs for v in views if os.path.exists(f'{vdir(v)}/{u}.npy')]
            BK = {'rep': torch.stack([rep(ld(f'{vdir(v)}/{u}.npy')) for u, v in rows]), 'rows': rows}; torch.save(BK, bank_f)
        TM = BK['rep'].mean(0); nz = lambda v: F.normalize(v - TM, dim=-1); TRn = nz(BK['rep'])
        R = {}
        for u, x in gen.items():
            sg = TRn @ nz(rep(x)); j = int(sg.argmax()); f = f'{a.tgt}/dino/{u}.npy'
            cr = float((TRn @ nz(rep(ld(f)))).max()) if os.path.exists(f) else float('nan')
            R[u] = {'gen_max_cos': round(float(sg[j]), 4), 'gen_nearest': list(BK['rows'][j]), 'real_max_cos': round(cr, 4)}
        gm, rm = np.mean([v['gen_max_cos'] for v in R.values()]), np.nanmean([v['real_max_cos'] for v in R.values()])
        json.dump({'mean_gen': gm, 'mean_real': rm, 'bank': len(BK['rows']), 'per': R}, open(f'{a.out}/copy_check.json', 'w'), indent=1)
        print(f'COPY 가장 가까운 학습 정답 토큰 cos ({len(BK["rows"])}개 중): 생성 {gm:.3f} | 그 물체 실제 토큰 {rm:.3f} | 복사면 1.000', flush=True)
    print('PRIOR_SAMPLE_DONE', flush=True)
