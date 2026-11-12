# 판정 2 (11-12): 갈래 끄기 · 바꿔 끼우기 — 디코더가 S · V · P를 실제로 쓰는가 (토큰만, 렌더 없음, MinD-3D++ venv, 실행 위치 $W/run)
#  조건 (피험자마다 test 104):
#    orig_re  원본 다시 계산 (흔들림 바닥)
#    S_mean · V_mean · P_mean  그 갈래를 그 피험자 train 평균으로
#    S_diff · S_same  Semantic Features를 다른 범주 물체 · 같은 범주 다른 물체 것으로 (같은 피험자 test)
#    V_diff · V_same  Variational Features를 같은 방식으로
#    P_sub    Perceptive 벡터를 다른 피험자(같은 물체) 것으로 (피험자 목록 다음 사람)
#  재는 것 (물체마다 → 평균):
#    d_out     1 − cos(원본 출력, 바꾼 출력)  (토큰마다 cos 평균)
#    d_gt      cos(바꾼 출력, 정답 토큰) − cos(원본 출력, 정답 토큰)
#    cat_follow · cat_keep  출력 토큰 범주(정답 토큰으로 학습한 선형 분류기)가 준 쪽 범주 · 자기 범주인 비율
#    shp_follow  출력 토큰에서 읽은 개체 형상(정답 토큰 → view_targets_shape 회귀)이 준 쪽(준 물체의 원본 출력에서 읽은 값)으로 간 비율
#                = (f_바꿈 − f_원본)·(f_준쪽 − f_원본) / |f_준쪽 − f_원본|², 0 = 안 따라감, 1 = 다 따라감
#  참고 거리: 원본 출력끼리 1 − cos — 같은 범주 다른 물체 / 다른 범주 / 같은 물체 다른 피험자
#  선형 분류기 · 회귀: 정답 토큰 [CLS ; 패치 평균] 2048차원, ridge, λ는 train 뒤 20%로 고름 (train만 사용)
#  --ex <uid 목록>: 첫 피험자에서 그 물체들의 S_diff · V_diff · V_same 토큰 저장 (GLB 그림용) → <out>/ex_tokens/<조건>/<uid>.npy, ex_donors.json
import os, sys, json, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True); p.add_argument('--subs', default='')
p.add_argument('--bs', type=int, default=8); p.add_argument('--seed', type=int, default=0); p.add_argument('--ex', default='')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval()
subs = a.subs.split(',') if a.subs else SUBS
rd = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
TR, TE = [k for k in rd('train') if k.split('/')[0] in SYN], rd('test')
cat = lambda k: SYN.index(k.split('/')[0])
uid = lambda k: k.split('/')[1]


def tfeat(t):                                           # [B, 1374, C] → [B, 2048]
    t = t.float(); return torch.cat([t[:, 0], t[:, 5:].mean(1)], -1)


def mcos(x, y):                                         # 토큰마다 cos 평균 [B]
    return torch.nn.functional.cosine_similarity(x.float(), y.float(), dim=-1).mean(-1)


# ---- 정답 토큰 → 선형 분류기 · 형상 회귀 (train만) ----
def gt_tok(k): return torch.from_numpy(np.load(f'{a.tgt}/dino/{uid(k)}.npy'))
Xtr = torch.stack([tfeat(gt_tok(k)[None])[0] for k in TR]).cuda()
_z = np.load(f'{a.tgt}/view_targets_shape.npz'); SH = {str(k).split('/')[-1]: (_z['shape'][i].mean(0), bool(_z['shape_ok'][i])) for i, k in enumerate(_z['keys'])}
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
nz = lambda X: (X - mu) / sd


def ridge(X, Y, lam):
    Xb = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1)
    return torch.linalg.solve(Xb.T @ Xb + lam * torch.eye(Xb.shape[1], device=X.device), Xb.T @ Y)


def apply(W, X): return torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1) @ W


Ytr_c = torch.nn.functional.one_hot(torch.tensor([cat(k) for k in TR]), len(SYN)).float().cuda()
okm = torch.tensor([SH.get(uid(k), (None, False))[1] for k in TR]).cuda()
Ytr_s = torch.from_numpy(np.stack([SH[uid(k)][0] if SH.get(uid(k), (None, False))[1] else np.zeros(4, np.float32) for k in TR])).float().cuda()
n_fit = int(len(TR) * 0.8); Xn = nz(Xtr); best = {}
for lam in [1e1, 1e2, 1e3, 1e4, 1e5]:
    Wc = ridge(Xn[:n_fit], Ytr_c[:n_fit], lam); acc = float((apply(Wc, Xn[n_fit:]).argmax(1) == Ytr_c[n_fit:].argmax(1)).float().mean())
    m = okm.clone(); m[:n_fit] = False; m2 = okm.clone(); m2[n_fit:] = False
    Ws = ridge(Xn[m2], Ytr_s[m2], lam); pr = apply(Ws, Xn[m]); r = float(np.mean([np.corrcoef(pr[:, j].cpu(), Ytr_s[m][:, j].cpu())[0, 1] for j in range(4)]))
    if acc > best.get('c', (-1,))[0]: best['c'] = (acc, lam)
    if r > best.get('s', (-9,))[0]: best['s'] = (r, lam)
Wc = ridge(Xn, Ytr_c, best['c'][1]); Ws = ridge(Xn[okm], Ytr_s[okm], best['s'][1])
f_cat = lambda t: apply(Wc, nz(tfeat(t))).argmax(1)
f_shp = lambda t: apply(Ws, nz(tfeat(t)))
Xte_gt = torch.stack([gt_tok(k) for k in TE]).cuda()
gt_acc = float((f_cat(Xte_gt).cpu() == torch.tensor([cat(k) for k in TE])).float().mean())
te_ok = torch.tensor([SH.get(uid(k), (None, False))[1] for k in TE]).cuda()
gs = f_shp(Xte_gt)[te_ok].cpu().numpy(); ys = np.stack([SH[uid(k)][0] for k, o in zip(TE, te_ok.tolist()) if o])
gt_r = float(np.mean([np.corrcoef(gs[:, j], ys[:, j])[0, 1] for j in range(4)]))
print(f'[probe] 범주 분류기 λ {best["c"][1]:g} (train 뒤 20% {best["c"][0]:.3f}) · 정답 test 토큰 정확도 {gt_acc:.3f} | '
      f'형상 회귀 λ {best["s"][1]:g} (train 뒤 20% r {best["s"][0]:.3f}) · 정답 test 토큰 r {gt_r:.3f}', flush=True)


def load_x(sub, ks):
    x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
    return x, torch.tensor(FRAMES).expand(len(ks), -1).cuda()


def run(sub, ks, **ov):
    x, fi = load_x(sub, ks); si = SUBS.index(sub)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return model(x, fi, torch.full((len(ks),), si, dtype=torch.long, device='cuda'), **ov)


# ---- 1. 피험자마다 train 평균 · test 원본 ----
TRM, ORIG = {}, {}
for sub in subs:
    sS = sV = sP = None; n = 0
    for i in range(0, len(TR), a.bs):
        o = run(sub, TR[i:i + a.bs]); b = len(TR[i:i + a.bs])
        sS = o['sem_feat'].float().sum(0) + (0 if sS is None else sS); sV = o['var_feat'].float().sum(0) + (0 if sV is None else sV)
        sP = o['perc_vec'].float().sum(0) + (0 if sP is None else sP); n += b
    TRM[sub] = (sS / n, sV / n, sP / n)
    S, V, P, T = [], [], [], []
    for i in range(0, len(TE), a.bs):
        o = run(sub, TE[i:i + a.bs])
        S.append(o['sem_feat'].float()); V.append(o['var_feat'].float()); P.append(o['perc_vec'].float()); T.append(o['cond_tokens'].half())
    ORIG[sub] = dict(S=torch.cat(S), V=torch.cat(V), P=torch.cat(P), T=torch.cat(T))
    print(f'[orig] {sub} train {n} · test {len(TE)}', flush=True)

# ---- 2. 준 쪽 물체 고르기 (고정 seed, 피험자 공통) ----
rng = np.random.default_rng(a.seed); C = np.array([cat(k) for k in TE])
don = {'diff': np.array([rng.choice(np.where(C != C[i])[0]) for i in range(len(TE))]),
       'same': np.array([rng.choice(np.where((C == C[i]) & (np.arange(len(TE)) != i))[0]) for i in range(len(TE))])}
gtT = Xte_gt
gt_cos0 = {s: torch.cat([mcos(ORIG[s]['T'][i:i + a.bs], gtT[i:i + a.bs]) for i in range(0, len(TE), a.bs)]) for s in subs}
res, ref = {}, {}
ex = [l.strip() for l in open(a.ex) if l.strip()] if a.ex else []
if ex: json.dump({'diff': {TE[i].split('/')[1]: TE[don['diff'][i]].split('/')[1] for i in range(len(TE))},
                  'same': {TE[i].split('/')[1]: TE[don['same'][i]].split('/')[1] for i in range(len(TE))}}, open(f'{a.out}/ex_donors.json', 'w'), indent=1)
CONDS = ['orig_re', 'S_mean', 'V_mean', 'P_mean', 'S_diff', 'S_same', 'V_diff', 'V_same', 'P_sub']
for si_, sub in enumerate(subs):
    Og = ORIG[sub]; psub = subs[(si_ + 1) % len(subs)]
    f0c = torch.cat([f_cat(Og['T'][i:i + a.bs]) for i in range(0, len(TE), a.bs)]).cpu()
    f0s = torch.cat([f_shp(Og['T'][i:i + a.bs]) for i in range(0, len(TE), a.bs)])
    fPs = torch.cat([f_shp(ORIG[psub]['T'][i:i + a.bs]) for i in range(0, len(TE), a.bs)])
    ref[sub] = {'same_cat_obj': float(np.mean([1 - mcos(Og['T'][[i]], Og['T'][[don['same'][i]]]).item() for i in range(len(TE))])),
                'diff_cat_obj': float(np.mean([1 - mcos(Og['T'][[i]], Og['T'][[don['diff'][i]]]).item() for i in range(len(TE))])),
                'same_obj_other_sub': float(np.mean([1 - mcos(Og['T'][[i]], ORIG[psub]['T'][[i]]).item() for i in range(len(TE))])),
                'orig_cat_acc': float((f0c == torch.tensor(C)).float().mean())}
    res[sub] = {}
    for cond in CONDS:
        m = {k: [] for k in ('d_out', 'd_gt', 'cat_follow', 'cat_keep', 'shp_follow')}
        for i in range(0, len(TE), a.bs):
            ks = TE[i:i + a.bs]; ix = np.arange(i, i + len(ks)); ov = {}; ixt = torch.from_numpy(ix).cuda()
            br, kind = (cond.split('_') + [''])[:2]
            if cond == 'S_mean': ov['sem_override'] = TRM[sub][0][None].expand(len(ks), -1, -1)
            if cond == 'V_mean': ov['var_override'] = TRM[sub][1][None].expand(len(ks), -1, -1)
            if cond == 'P_mean': ov['perceptive_override'] = TRM[sub][2][None].expand(len(ks), -1)
            if cond in ('S_diff', 'S_same'): ov['sem_override'] = Og['S'][torch.from_numpy(don[kind][ix]).cuda()]
            if cond in ('V_diff', 'V_same'): ov['var_override'] = Og['V'][torch.from_numpy(don[kind][ix]).cuda()]
            if cond == 'P_sub': ov['perceptive_override'] = ORIG[psub]['P'][ixt]
            T = run(sub, ks, **ov)['cond_tokens']
            m['d_out'].append(1 - mcos(T, Og['T'][ixt])); m['d_gt'].append(mcos(T, gtT[ixt]) - gt_cos0[sub][ixt])
            pc = f_cat(T).cpu(); m['cat_keep'].append((pc == torch.from_numpy(C[ix])).float())
            if kind in ('diff', 'same'):
                m['cat_follow'].append((pc == torch.from_numpy(C[don[kind][ix]])).float())
            fs = f_shp(T); f0 = f0s[ixt]
            fd = f0s[torch.from_numpy(don[kind][ix]).cuda()] if kind in ('diff', 'same') else (fPs[ixt] if cond == 'P_sub' else None)
            if fd is not None:
                dd = fd - f0; m['shp_follow'].append(((fs - f0) * dd).sum(1) / (dd * dd).sum(1).clamp_min(1e-8))
            if ex and sub == subs[0] and cond in ('S_diff', 'V_diff', 'V_same'):
                for j, k in enumerate(ks):
                    if uid(k) in ex:
                        os.makedirs(f'{a.out}/ex_tokens/{cond}', exist_ok=True); np.save(f'{a.out}/ex_tokens/{cond}/{uid(k)}.npy', T[j].float().half().cpu().numpy())
        res[sub][cond] = {k: (float(torch.cat(v).float().median()) if k == 'shp_follow' else float(torch.cat(v).float().mean())) for k, v in m.items() if v}
        print(f'[{sub}] {cond:8s} ' + ' '.join(f'{k} {v:.4f}' for k, v in res[sub][cond].items()), flush=True)
    print(f'[{sub}] 참고 ' + ' '.join(f'{k} {v:.4f}' for k, v in ref[sub].items()), flush=True)

avg = {c: {k: float(np.mean([res[s][c][k] for s in subs])) for k in res[subs[0]][c]} for c in CONDS}
aref = {k: float(np.mean([ref[s][k] for s in subs])) for k in ref[subs[0]]}
json.dump({'ckpt': a.ckpt, 'subs': subs, 'probe': {'cat_lam': best['c'][1], 'cat_gt_test_acc': gt_acc, 'shp_lam': best['s'][1], 'shp_gt_test_r': gt_r},
           'per_sub': res, 'ref': ref, 'avg': avg, 'avg_ref': aref}, open(f'{a.out}/swap.json', 'w'), indent=1)
for c in CONDS: print(f'AVG {c:8s} ' + ' '.join(f'{k} {v:.4f}' for k, v in avg[c].items()), flush=True)
print('AVG 참고 ' + ' '.join(f'{k} {v:.4f}' for k, v in aref.items()), flush=True)
print('SWAP_DONE', flush=True)
