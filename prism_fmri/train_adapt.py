# -*- coding: utf-8 -*-
# 11-20 Perceptive 1단계 (학습판): 학습에 없던 피험자 B를 학습된 모델에 붙이기 — 모델 전체(ViT · Semantic · Variational · Decoder)는 고정하고, B용 가벼운 정렬층만 물체 N개로 학습
#  정렬층 (고정 ViT 토큰 [257, 1024] 위): ① B 자신의 통계로 표준화 → ② 모든 위치 공통 Linear 1024 → 1024 (단위행렬에서 시작) + 위치별 bias (+ --rank r > 0이면 위치별 저랭크 보정, 0에서 시작) → ③ A의 통계로 되돌림 → A 갈래
#   학습 전(스텝 0) = 통계만 맞춘 상태 (align_infer.py --fmap stat과 같음)
#  손실: 본 학습과 같은 토큰 MSE + (1 − cos) + 범주 CE(CLIP 텍스트 프로토타입). 정답은 자극에서 나오므로 다른 피험자의 fMRI 짝이 필요 없음 (B의 "자극 – fMRI" N개만)
#  고르기: N개 중 뒤 20 % 물체를 빼 두고, 그 물체의 손실이 가장 낮은 스텝의 정렬층으로 test 104개를 잼 (test는 고르는 데 안 씀). 물체 = core_train 고정 순서(seed 0) 앞 N개 (align_infer.py와 같음)
#  출력: <out>/adapt.json (쌍 · N별: 스텝 0 / 고른 스텝의 test 범주 정확도 · 토큰 cos, A 자신의 값) + --save "N,…"이면 <out>/n<N>/<B>_as<A 번호>_keep/{tokens, components.npz} · adapt_<A>_<B>_n<N>.pt
# 사용 (실행 위치 $W/run): python train_adapt.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --ckpt $W/prism_2s/ausvp/model_final.pt --subs sub-0001 --fmri_subs sub-0002 --ns 100,300,1300 --out $W/prism_align/adapt
import os, sys, json, time, math, copy, argparse, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True); p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
p.add_argument('--subs', required=True); p.add_argument('--fmri_subs', required=True); p.add_argument('--ns', default='100,300,1300')
p.add_argument('--steps', type=int, default=1500); p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--bs', type=int, default=16); p.add_argument('--wd', type=float, default=0.01)
p.add_argument('--rank', type=int, default=0); p.add_argument('--val_every', type=int, default=100); p.add_argument('--save', default='')
a = p.parse_args()
dev = 'cuda'; t0 = time.time(); ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8)); T = len(FRAMES)
model = build_from_ckpt(ck, a.repo).cuda().eval().requires_grad_(False); ENC = model.encode_frames; model.encode_frames = lambda f, s: f
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
has = lambda k: os.path.exists(f'{a.tgt}/dino/{k.split("/")[1]}.npy')
_tr = lst('train'); TRP = [k for k in (_tr[i] for i in torch.randperm(len(_tr), generator=torch.Generator().manual_seed(0)).tolist()) if has(k)]; TE = [k for k in lst('test') if has(k)]
lab = lambda keys: torch.tensor([SYN.index(k.split('/')[0]) for k in keys], device=dev); SAVE = {s for s in a.save.split(',') if s}; os.makedirs(a.out, exist_ok=True)
tgt = lambda keys: torch.from_numpy(np.stack([np.load(f'{a.tgt}/dino/{k.split("/")[1]}.npy') for k in keys])).to(dev)   # [n, 1374, 1024]
NSL = [int(v) for v in a.ns.split(',')]; NM = max(NSL)


@torch.no_grad()
def toks(s, keys, si):                                                      # 고정 ViT 토큰 [n, T, 257, D] fp16
    o = []
    for i in range(0, len(keys), 32):
        x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in keys[i:i + 32]])).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16): o.append(ENC(x, torch.full((len(x),), si, dtype=torch.long, device=dev)).half())
    return torch.cat(o)


class Adapt(nn.Module):
    def __init__(s, xb, xa, rank=0):
        super().__init__(); st = lambda x: (lambda f: (f.mean((0, 1)), f.std((0, 1)) + 1e-6))(x.float()); D = xb.shape[-1]
        for n_, v in zip(('mb', 'sb', 'ma', 'sa'), st(xb) + st(xa)): s.register_buffer(n_, v)                  # [257, D]
        s.W = nn.Parameter(torch.eye(D)); s.b = nn.Parameter(torch.zeros(257, D)); s.rank = rank
        if rank: s.U = nn.Parameter(torch.randn(257, D, rank) * 0.02); s.V = nn.Parameter(torch.zeros(257, rank, D))
    def forward(s, x):                                                      # [n, T, 257, D]
        z = (x.float() - s.mb) / s.sb; y = z @ s.W + s.b
        if s.rank: y = y + torch.einsum('ntpd,pdr,pre->ntpe', z, s.U, s.V)
        return y * s.sa + s.ma


def fwd(ad, x, si):
    tok = ad(x) if ad is not None else x.float(); fi = torch.tensor(FRAMES, device=dev).expand(len(tok), -1)
    with torch.autocast('cuda', dtype=torch.bfloat16): return model(tok, fi, torch.full((len(tok),), si, dtype=torch.long, device=dev))
def loss_of(out, y, t):
    o = out['cond_tokens'].float(); lg = out['sem_clip'].float() @ model.prototypes.t() / 0.07
    return F.mse_loss(o, t.float()) + (1 - F.cosine_similarity(o, t.float(), dim=-1)).mean() + F.cross_entropy(lg, y), lg
@torch.no_grad()
def evaluate(ad, X, keys, si, od=None):                                     # 범주 정확도 · 토큰 cos · 손실
    y = lab(keys); hit, cs, ls, comp = [], [], [], {k: [] for k in ('sem', 'var', 'perc', 'cls_score', 'clip', 'vt')}
    if od: os.makedirs(f'{od}/tokens', exist_ok=True)
    for i in range(0, len(keys), 16):
        out = fwd(ad, X[i:i + 16], si); t = tgt(keys[i:i + 16]); l, lg = loss_of(out, y[i:i + 16], t)
        hit.append((lg.argmax(-1) == y[i:i + 16]).float().cpu()); cs.append(F.cosine_similarity(out['cond_tokens'].float(), t.float(), dim=-1).mean(-1).cpu()); ls.append(float(l) * len(t))
        if od:
            for j, k in enumerate(keys[i:i + 16]): np.save(f'{od}/tokens/{k.split("/")[1]}.npy', out['cond_tokens'][j].float().half().cpu().numpy())
            comp['sem'].append(out['sem_vec'].float().cpu()); comp['var'].append(out['var_vec'].float().cpu()); comp['perc'].append(out['perc_vec'].float().cpu()); comp['cls_score'].append((lg * 0.07).cpu()); comp['clip'].append(out['sem_clip'].float().cpu())
            if 'vt' in out: comp['vt'].append(out['vt'].float().mean(1).cpu())
    if od: np.savez(f'{od}/components.npz', name=np.array(keys), cls=y.cpu().numpy(), classes=np.array(SYN), cls_pred=torch.cat(comp['cls_score']).argmax(-1).numpy(), **{k: torch.cat(v).numpy() for k, v in comp.items() if v})
    return round(float(torch.cat(hit).mean()), 4), round(float(torch.cat(cs).mean()), 4), sum(ls) / len(keys)


R = {'ckpt': a.ckpt, 'steps': a.steps, 'lr': a.lr, 'rank': a.rank, 'chance': round(1 / len(SYN), 4), 'pairs': {}}
for A in a.subs.split(','):
    si = SUBS.index(A); FA = toks(A, TRP[:NM], si); FAte = toks(A, TE, si); own = evaluate(None, FAte, TE, si)[:2]; del FAte
    print(f'[{A}] 자기 test 범주 {own[0]:.3f} · 토큰 cos {own[1]:.3f} | {time.time() - t0:.0f}s', flush=True)
    for B in a.fmri_subs.split(','):
        if B == A: continue
        FB = toks(B, TRP[:NM], si); FBte = toks(B, TE, si); r = {'own_A': own, 'raw': evaluate(None, FBte, TE, si)[:2], 'n': {}}
        print(f'[{A} ← {B}] 그대로 넣음: 범주 {r["raw"][0]:.3f} · cos {r["raw"][1]:.3f}', flush=True)
        for N in NSL:
            nf = max(int(N * 0.8), 1); trk, hk = TRP[:nf], TRP[nf:N]; ytr = lab(trk); Ttr = tgt(trk).half()
            torch.manual_seed(0); ad = Adapt(FB[:nf], FA, a.rank).to(dev); opt = torch.optim.AdamW(ad.parameters(), lr=a.lr, weight_decay=a.wd)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
            s0 = evaluate(ad, FBte, TE, si)[:2]; best = (evaluate(ad, FB[nf:N], hk, si)[2], 0, copy.deepcopy(ad.state_dict())); curve = [(0, round(best[0], 4))]
            for step in range(1, a.steps + 1):
                idx = torch.randint(0, nf, (a.bs,), device=dev); l, _ = loss_of(fwd(ad, FB[idx], si), ytr[idx], Ttr[idx])
                opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0); opt.step(); sched.step()
                if step % a.val_every == 0:
                    h = evaluate(ad, FB[nf:N], hk, si)[2]; curve.append((step, round(h, 4)))
                    if h < best[0]: best = (h, step, copy.deepcopy(ad.state_dict()))
            ad.load_state_dict(best[2]); od = f'{a.out}/n{N}/{B}_as{A[-2:]}_keep' if str(N) in SAVE else None
            te = evaluate(ad, FBte, TE, si, od)[:2]
            if od: torch.save({'state': best[2], 'A': A, 'B': B, 'N': N, 'step': best[1]}, f'{a.out}/adapt_{A[-2:]}_{B[-2:]}_n{N}.pt')
            r['n'][str(N)] = {'step0': s0, 'test': te, 'best_step': best[1], 'hold_loss': round(best[0], 4), 'hold_curve': curve, 'fit_n': nf, 'hold_n': len(hk)}
            print(f'[{A} ← {B}] N {N} | 통계만(스텝 0) 범주 {s0[0]:.3f} · cos {s0[1]:.3f} → 학습 뒤 범주 {te[0]:.3f} · cos {te[1]:.3f} | 고른 스텝 {best[1]} (검증 손실 {curve[0][1]:.3f} → {best[0]:.3f}) | {time.time() - t0:.0f}s', flush=True)
            del ad, opt, Ttr; torch.cuda.empty_cache()
        R['pairs'][f'{A}<-{B}'] = r; json.dump(R, open(f'{a.out}/adapt.json', 'w'), ensure_ascii=False, indent=1); del FB, FBte; torch.cuda.empty_cache()
    del FA; torch.cuda.empty_cache()
S = {str(N): {k: [round(float(np.mean([r['n'][str(N)][k][j] for r in R['pairs'].values()])), 4) for j in (0, 1)] for k in ('step0', 'test')} for N in NSL}
R['mean'] = S; R['own_mean'] = [round(float(np.mean([r['own_A'][j] for r in R['pairs'].values()])), 4) for j in (0, 1)]; R['raw_mean'] = [round(float(np.mean([r['raw'][j] for r in R['pairs'].values()])), 4) for j in (0, 1)]
json.dump(R, open(f'{a.out}/adapt.json', 'w'), ensure_ascii=False, indent=1); print('쌍 평균 [범주, cos]', json.dumps(S), '| 자기', R['own_mean'], '| 그대로', R['raw_mean'], flush=True); print('ADAPT_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
