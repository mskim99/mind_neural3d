# -*- coding: utf-8 -*-
# 11-20 Perceptive 1단계 (새 피험자 붙이기): 학습에 없던 피험자 B의 fMRI를 학습된 피험자 A 쪽으로 옮기는 선형 사상(기능 정렬)을 같은 물체 N개 짝으로 맞춘 뒤,
#  A의 갈래(고정, 추가 학습 없음)로 그대로 추론 — N을 바꿔 가며 범주 정확도를 잼 (N-shot 곡선). 진단 diag_align.py(선형 분류기)의 실제 모델판
#  --space input : 사상을 모델 입력(in_norm 뒤 평면 지도의 0 아닌 픽셀)에서 맞춤 → 고정 ViT → A 갈래
#          feat  : 고정 ViT 출력 토큰 [257, 1024]에서 맞춤 (ViT는 A의 것을 B에도 그대로) → A 갈래 (feat_norm은 A의 값)
#  --fmap full : 전체를 한 벡터로 본 커널 릿지 (--fit frame = (물체, 프레임)이 표본 / mean = 물체의 프레임 평균이 표본, 사상은 프레임마다 적용)
#         pos  : (feat만) 토큰 위치마다 1024 → 1024 릿지 (표본 = 물체 × 프레임). 위치를 섞지 않는 가벼운 사상
#         stat : (feat만, 짝이 필요 없음) B의 토큰을 위치 · 차원마다 B 자신의 평균 · 표준편차(물체 N개)로 표준화한 뒤 A의 평균 · 표준편차로 되돌림 — 통계만 맞추는 기준
#  사상은 프레임별 평균을 뺀 값끼리 맞추고 A의 평균을 더해 돌려줌. 짝에는 같은 자극을 본 fMRI만 있으면 됨
#  λ · 배율 고르기 (test 안 씀): 짝의 앞 80 %로 맞추고 뒤 20 % 물체를 옮겨 A 갈래에 넣었을 때의 범주 정확도가 가장 높은 것 (자극의 범주는 아는 값).
#   배율 g1 = 그대로, gm = 릿지가 줄인 변화 크기를 같은 20 %에서 A의 실제 크기에 맞춤. 고른 뒤 N개 전부로 다시 맞춰 test 104개에 적용
#   (20 %에서 A 실제 패턴과의 상관도 적어 두지만 고르는 데는 안 씀: 제시 순서가 같은 피험자끼리는 run · 순서 성분이 맞아서 자극과 무관하게 높게 나옴 — 순서 묶음 {1,6,7,8} · {2,3,5})
#  N = 0: 정렬 없이 B를 그대로 A 갈래에 (기준). 짝 물체 = core_train 목록 고정 순서(seed 0) 앞 N개
#  출력: <out>/align.json + --save "N,…"이면 <out>/n<N>/<B>_as<A 번호>_keep/{tokens, components.npz} (infer_prism.py와 같은 형식)
# 사용 (실행 위치 $W/run): python align_infer.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_2s/ausvp/model_final.pt --subs sub-0001,sub-0006 --fmri_subs sub-0002,sub-0003 --space feat --fmap pos --out $W/prism_align/feat_pos
import os, sys, json, time, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
p.add_argument('--subs', required=True); p.add_argument('--fmri_subs', required=True); p.add_argument('--ns', default='0,50,100,300,600,1300')
p.add_argument('--space', default='feat', choices=['input', 'feat']); p.add_argument('--fmap', default='full', choices=['full', 'pos', 'stat', 'region']); p.add_argument('--fit', default='mean', choices=['frame', 'mean'])
p.add_argument('--lams', default='0.01,0.03,0.1,0.3,1,3,10'); p.add_argument('--save', default=''); p.add_argument('--bs', type=int, default=16)
p.add_argument('--rw_common', type=int, default=0, choices=[0, 1])   # 11-21 P-2: 영역 가중 피험자별 조정 d를 0으로 (새 피험자는 공통 가중만)
p.add_argument('--save_tok', type=int, default=1, choices=[0, 1])    # 11-23: 0이면 components.npz만 (조건 토큰 저장 안 함)
p.add_argument('--tgt_pool', type=int, default=0, choices=[0, 1]); p.add_argument('--pool_lam', type=float, default=0.1)   # 11-23: 목표 = A + (다른 학습 피험자를 A로 옮긴 값)의 평균 — 2겹 밖 예측, 새 피험자 데이터는 안 씀
p.add_argument('--cv', default='')   # 11-23: train을 안 본 새 피험자(AP 9번)용. "K,…" — test 물체를 K겹으로 나눠, 한 겹을 뺀 나머지 물체 짝으로 변환을 맞추고 뺀 겹에 적용 (모든 test 물체가 자기를 안 쓴 변환으로 옮겨짐)
a = p.parse_args(); assert a.fmap == 'full' or a.space == ('input' if a.fmap == 'region' else 'feat')
dev = 'cuda'; t0 = time.time(); ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8)); T = len(FRAMES)
model = build_from_ckpt(ck, a.repo).cuda().eval(); ENC = model.encode_frames; model.encode_frames = lambda f, s: f     # 모델에는 ViT 토큰을 바로 넣음
if a.rw_common:
    assert hasattr(model, 'rw'), '--rw_common: 영역 가중 없는 모델'
    with torch.no_grad(): model.rw.d_s.zero_(); model.rw.d_v.zero_()
    print('[rw_common] 영역 가중 피험자별 조정 0 → 공통 가중만', flush=True)
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
TR, TE = lst('train'), lst('test'); TRP = [TR[i] for i in torch.randperm(len(TR), generator=torch.Generator().manual_seed(0)).tolist()]
lab = lambda keys: np.array([SYN.index(k.split('/')[0]) for k in keys]); SAVE = {s for s in a.save.split(',') if s}; GRP = lambda s: 'a' if s[-1] in '1678' else 'b'
os.makedirs(a.out, exist_ok=True)


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m
def raw(s, keys): return torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in keys]))
M = (raw(a.subs.split(',')[0], TE[:4]) != 0).any(0).any(0).flatten().to(dev); P = int(M.sum())
if a.fmap == 'region': LABM = torch.as_tensor(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mmp_256_regions.npy'))).long().flatten().to(dev)[M]; print(f'[region] 픽셀 {P} · 영역 {len(LABM.unique())} (라벨 0 = 영역 밖 한 묶음)', flush=True)


@torch.no_grad()
def vit(x, si):                                                             # [n, T, 256, 256] → 토큰 [n, T, 257·D] fp16
    with torch.autocast('cuda', dtype=torch.bfloat16): return ENC(x, torch.full((len(x),), si, dtype=torch.long, device=dev)).flatten(2).half()
@torch.no_grad()
def feat(s, keys, si):                                                      # 사상을 맞추는 공간의 값 [n, T, F]
    o = []
    for i in range(0, len(keys), 32):
        x = raw(s, keys[i:i + 32]).to(dev); o.append(norm_in(x).flatten(2)[:, :, M] if a.space == 'input' else vit(x, si))
    return torch.cat(o)
@torch.no_grad()
def to_tok(z, si):                                                          # 옮긴 값 [n, T, F] → 토큰 [n, T, 257, D]
    if a.space == 'input':
        o = torch.zeros(*z.shape[:2], 256 * 256, device=dev); o[:, :, M] = z.float(); z = vit(o.view(*z.shape[:2], 256, 256), si)
    return z.view(*z.shape[:2], 257, -1)


@torch.no_grad()
def run(zf, si, keys, od=None):                                             # zf(i0, i1) → [n, T, F] 옮긴 값. 범주 정확도 (+ 저장)
    y = lab(keys); pr, comp = [], {k: [] for k in ('sem', 'var', 'perc', 'cls_score', 'clip', 'vt')}
    if od: os.makedirs(f'{od}/tokens' if a.save_tok else od, exist_ok=True)
    for i in range(0, len(keys), a.bs):
        tok = to_tok(zf(i, min(i + a.bs, len(keys))), si); fi = torch.tensor(FRAMES, device=dev).expand(len(tok), -1)
        with torch.autocast('cuda', dtype=torch.bfloat16): out = model(tok, fi, torch.full((len(tok),), si, dtype=torch.long, device=dev))
        sc = (out['sem_clip'].float() @ model.prototypes.t()).cpu(); pr.append(sc.argmax(-1))
        if od:
            for j, k in enumerate(keys[i:i + a.bs] if a.save_tok else []): np.save(f'{od}/tokens/{k.split("/")[1]}.npy', out['cond_tokens'][j].float().half().cpu().numpy())
            comp['sem'].append(out['sem_vec'].float().cpu()); comp['var'].append(out['var_vec'].float().cpu()); comp['perc'].append(out['perc_vec'].float().cpu()); comp['cls_score'].append(sc); comp['clip'].append(out['sem_clip'].float().cpu())
            if 'vt' in out: comp['vt'].append(out['vt'].float().mean(1).cpu())
    pr = torch.cat(pr).numpy()
    if od: np.savez(f'{od}/components.npz', name=np.array(keys), cls=y, classes=np.array(SYN), cls_pred=pr, **{k: torch.cat(v).numpy() for k, v in comp.items() if v})
    return float((pr == y).mean())


def ksolve(K, Y, lam): return torch.linalg.solve(K.double() + lam * K.diagonal().mean().double() * torch.eye(len(K), device=dev, dtype=torch.float64), Y.double()).float()
class Map:                                                                  # B → A 사상 (표본 xb · xa [n, T, F] fp16/32, 평균 mub · mua [1, T, F])
    def __init__(s, xb, xa, mub, mua, lam):
        s.mub, s.mua, s.g = mub, mua, 1.0
        if a.fmap == 'full':
            if a.fit == 'frame': db, da = (xb.float() - mub).flatten(0, 1), (xa.float() - mua).flatten(0, 1)
            else: db, da = (torch.cat([(x[i:i + 64].float() - mu).mean(1) for i in range(0, len(x), 64)]) for x, mu in ((xb, mub), (xa, mua)))   # 64개씩 (fp16 → fp32 사본을 한꺼번에 안 만듦)
            s.db = db; s.Wd = ksolve(db @ db.t(), da, lam)
        elif a.fmap == 'region':                                             # 11-23: 뇌 영역마다 따로 (영역 r의 픽셀 → 같은 영역 r의 픽셀). 표본 = 물체의 프레임 평균
            db, da = (torch.cat([(x[i:i + 64].float() - mu).mean(1) for i in range(0, len(x), 64)]) for x, mu in ((xb, mub), (xa, mua))); s.parts = []
            for r in LABM.unique().tolist():
                ix = (LABM == r).nonzero(as_tuple=True)[0]; d = db[:, ix]; s.parts.append((ix, d, ksolve(d @ d.t(), da[:, ix], lam)))
        else:
            D = xb.shape[-1] // 257; G = torch.zeros(257, D, D, device=dev); C = torch.zeros(257, D, D, device=dev)
            for i in range(0, len(xb), 64):
                db = (xb[i:i + 64].float() - mub).view(-1, 257, D); da = (xa[i:i + 64].float() - mua).view(-1, 257, D)
                G += torch.einsum('spd,spe->pde', db, db); C += torch.einsum('spd,spe->pde', db, da)
            tr = G.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
            s.W = torch.linalg.solve(G + lam * tr * torch.eye(D, device=dev), C); s.D = D
    def dev_(s, x):                                                         # [n, T, F] → 옮긴 변화량 [n, T, F]
        d = x.float() - s.mub
        if a.fmap == 'full': return (d @ s.db.t()) @ s.Wd
        if a.fmap == 'region':
            o = torch.zeros_like(d)
            for ix, db, Wd in s.parts: o[..., ix] = (d[..., ix] @ db.t()) @ Wd
            return o
        return torch.einsum('ntpd,pde->ntpe', d.view(*d.shape[:2], 257, s.D), s.W).flatten(2)
    def __call__(s, x): return s.mua + s.g * s.dev_(x)


def fmean(x): return sum(x[i:i + 64].float().sum(0, keepdim=True) for i in range(0, len(x), 64)) / len(x)                       # [1, T, F] (64개씩)
def rcorr(A, B): A = A - A.mean(-1, keepdim=True); B = B - B.mean(-1, keepdim=True); return float((F.normalize(A, dim=-1) * F.normalize(B, dim=-1)).sum(-1).mean())
def pick_fit(xb, xa, keys, si):                                             # 짝 [N, T, F] → 앞 80 %로 맞추고 뒤 20 %에서 λ · 배율을 고른 뒤 전부로 다시 맞춘 사상
    N = len(xb); nf = max(int(N * 0.8), 1); hk = keys[nf:]; best = None; mua, mub = fmean(xa[:nf]), fmean(xb[:nf])
    for lam in LAMS:
        m = Map(xb[:nf], xa[:nf], mub, mua, lam); ph = torch.cat([m.dev_(xb[i:i + 16]) for i in range(nf, N, 16)]); ah = xa[nf:N].float() - mua
        g = float(ah.pow(2).mean().sqrt() / ph.pow(2).mean().sqrt().clamp_min(1e-8)); del ph, ah
        for tag, gg in (('g1', 1.0), ('gm', g)):
            m.g = gg; h = run(lambda i, j: m(xb[nf + i:nf + j]), si, hk)
            if best is None or h > best[0]: best = (h, lam, tag, gg)
        del m; torch.cuda.empty_cache()
    h, lam, tag, gg = best; m = Map(xb, xa, fmean(xb), fmean(xa), lam); m.g = gg; return m, {'lam': lam, 'gain': tag, 'hold_acc': round(h, 4)}


def cv_run(A, si):
    FAte = feat(A, TE, si); own = run(lambda i, j: FAte[i:j], si, TE); print(f'[{A}] 자기 test 범주 정확도 {own:.3f} | {time.time() - t0:.0f}s', flush=True)
    for B in a.fmri_subs.split(','):
        FBte = feat(B, TE, si); r = {'own_A': round(own, 4), 'same_order': GRP(A) == GRP(B), 'n': {}}
        od = f'{a.out}/n0/{B}_as{A[-2:]}_keep' if '0' in SAVE else None
        acc = run(lambda i, j: FBte[i:j], si, TE, od); r['n']['0'] = {'test': round(acc, 4)}; print(f'[{A} ← {B}] 짝 0 (그대로) 범주 {acc:.3f}', flush=True)
        for K in [int(v) for v in a.cv.split(',')]:
            pm = torch.randperm(len(TE), generator=torch.Generator().manual_seed(0)); Z = torch.zeros_like(FBte); info = []
            for k in range(K):
                te_ix = pm[k::K]; fi = torch.cat([pm[q::K] for q in range(K) if q != k]); fd = fi.to(dev)
                m, inf = pick_fit(FBte[fd], FAte[fd], [TE[i] for i in fi.tolist()], si); info.append(inf)
                Z[te_ix.to(dev)] = m(FBte[te_ix.to(dev)]).half(); del m; torch.cuda.empty_cache()
            n = len(TE) - len(pm[0::K]); od = f'{a.out}/n{n}/{B}_as{A[-2:]}_keep' if str(K) in SAVE else None
            acc = run(lambda i, j: Z[i:j], si, TE, od); r['n'][str(n)] = {'test': round(acc, 4), 'folds': K, 'fit': info}
            print(f'[{A} ← {B}] {K}겹 (짝 약 {n}개) | test 범주 {acc:.3f} | {time.time() - t0:.0f}s', flush=True); del Z
        R['pairs'][f'{A}<-{B}'] = r; json.dump(R, open(f'{a.out}/align.json', 'w'), ensure_ascii=False, indent=1); del FBte; torch.cuda.empty_cache()


NSL = [int(v) for v in a.ns.split(',')]; LAMS = [float(v) for v in a.lams.split(',')]; NM = max(NSL)
@torch.no_grad()
def pool_target(A, FA):                                                     # [NM, T, F] fp16: A와, 다른 학습 피험자를 A로 옮긴 값의 평균
    oth = [s for s in SUBS if s != A and s not in a.fmri_subs.split(',')]; acc = FA.clone(); h = NM // 2
    for j in oth:
        Fj = feat(j, TRP[:NM], SUBS.index(j))
        for f0, f1, a0, a1 in ((h, NM, 0, h), (0, h, h, NM)):                 # 절반으로 맞추고 나머지 절반에 적용
            m = Map(Fj[f0:f1], FA[f0:f1], fmean(Fj[f0:f1]), fmean(FA[f0:f1]), a.pool_lam)
            for i in range(a0, a1, 32): e = min(i + 32, a1); acc[i:e] = (acc[i:e].float() + m(Fj[i:e])).half()
            del m; torch.cuda.empty_cache()
        del Fj; torch.cuda.empty_cache()
    for i in range(0, NM, 64): acc[i:i + 64] = (acc[i:i + 64].float() / (len(oth) + 1)).half()
    print(f'[{A}] 목표 평균 (다른 학습 피험자 {len(oth)}명을 옮겨 더함) | {time.time() - t0:.0f}s', flush=True); return acc


R = {'ckpt': a.ckpt, 'space': a.space, 'fmap': a.fmap, 'fit': a.fit, 'tgt_pool': a.tgt_pool, 'chance': round(1 / len(SYN), 4), 'pairs': {}}
for A in a.subs.split(','):
    si = SUBS.index(A)
    if a.cv: cv_run(A, si); continue
    FA = feat(A, TRP[:NM], si); FAte = feat(A, TE, si)
    own = run(lambda i, j: FAte[i:j], si, TE); print(f'[{A}] 자기 test 범주 정확도 {own:.3f} | {time.time() - t0:.0f}s', flush=True); del FAte
    if a.tgt_pool: FA = pool_target(A, FA); torch.cuda.empty_cache()
    for B in a.fmri_subs.split(','):
        if B == A: continue
        FB = feat(B, TRP[:NM], si); FBte = feat(B, TE, si); r = {'own_A': round(own, 4), 'same_order': GRP(A) == GRP(B), 'n': {}}
        for N in NSL:
            od = f'{a.out}/n{N}/{B}_as{A[-2:]}_keep' if str(N) in SAVE else None
            if N == 0:
                acc = run(lambda i, j: FBte[i:j], si, TE, od); r['n']['0'] = {'test': round(acc, 4)}; print(f'[{A} ← {B}] N 0 (정렬 없음) 범주 {acc:.3f}', flush=True); continue
            if a.fmap == 'stat':
                st = lambda x: (lambda f: (f.mean((0, 1), keepdim=True), f.std((0, 1), keepdim=True) + 1e-6))(x.float())
                (ma, sa), (mb, sb) = st(FA), st(FB[:N]); acc = run(lambda i, j: (FBte[i:j].float() - mb) / sb * sa + ma, si, TE, od)
                r['n'][str(N)] = {'test': round(acc, 4)}; print(f'[{A} ← {B}] N {N} | test 범주 {acc:.3f} (통계만 맞춤) | {time.time() - t0:.0f}s', flush=True); del ma, sa, mb, sb; continue
            xa, xb = FA[:N], FB[:N]; nf = max(int(N * 0.8), 1); hk = TRP[nf:N]; best = None; tab = {}
            mua, mub = fmean(xa[:nf]), fmean(xb[:nf])
            for lam in LAMS:                                                # 앞 80 %로 맞추고 뒤 20 % 물체에서 고름
                m = Map(xb[:nf], xa[:nf], mub, mua, lam); ph = torch.cat([m.dev_(xb[i:i + 16]) for i in range(nf, N, 16)]); ah = xa[nf:N].float() - mua
                c = rcorr(ph, ah); g = float(ah.pow(2).mean().sqrt() / ph.pow(2).mean().sqrt().clamp_min(1e-8)); del ph, ah
                for tag, gg in (('g1', 1.0), ('gm', g)):
                    m.g = gg; h = run(lambda i, j: m(xb[nf + i:nf + j]), si, hk); tab[f'{lam:g}_{tag}'] = round(h, 4)
                    if best is None or h > best[0]: best = (h, lam, tag, gg, c)
                del m; torch.cuda.empty_cache()
            h, lam, tag, gg, c = best
            m = Map(xb, xa, fmean(xb), fmean(xa), lam); m.g = gg
            acc = run(lambda i, j: m(FBte[i:j]), si, TE, od)
            r['n'][str(N)] = {'test': round(acc, 4), 'lam': lam, 'gain': tag, 'g': round(gg, 3), 'hold_acc': round(h, 4), 'hold_n': len(hk), 'corr20': round(c, 4), 'hold_tab': tab}
            print(f'[{A} ← {B}] N {N} | test 범주 {acc:.3f} | 고른 λ {lam:g} · {tag}({gg:.2f}) · 20 % 범주 {h:.3f} · 상관 {c:.3f} | {time.time() - t0:.0f}s', flush=True)
            del m; torch.cuda.empty_cache()
        R['pairs'][f'{A}<-{B}'] = r; json.dump(R, open(f'{a.out}/align.json', 'w'), ensure_ascii=False, indent=1); del FB, FBte; torch.cuda.empty_cache()
    del FA; torch.cuda.empty_cache()
S = {}
for N in ([int(k) for k in next(iter(R['pairs'].values()))['n']] if a.cv else NSL):
    for nm, sel in (('all', lambda r: True), ('same_order', lambda r: r['same_order']), ('diff_order', lambda r: not r['same_order'])):
        v = [r['n'][str(N)]['test'] for r in R['pairs'].values() if sel(r)]
        if v: S.setdefault(nm, {})[str(N)] = round(float(np.mean(v)), 4)
R['mean'] = S; R['own_mean'] = round(float(np.mean([r['own_A'] for r in R['pairs'].values()])), 4)
json.dump(R, open(f'{a.out}/align.json', 'w'), ensure_ascii=False, indent=1); print('쌍 평균', json.dumps(S, ensure_ascii=False), '| 자기', R['own_mean'], flush=True); print('ALIGN_INFER_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
