# -*- coding: utf-8 -*-
# 11-28 V-9: Variational 구조 후보의 정답이 범주를 빼고도 fMRI에서 읽히나 (학습 없음, 커널 릿지 — v5_read.py와 같은 입력 · λ 고르기)
#  정답 (물체마다, 정답 범주의 train 평균을 뺀 값, 차원마다 train 표준편차로 나눔):
#   blur8_f0 · blur16_f0 · blur8_avg · blur16_avg : 흐릿한 이미지 RGB (v9_feat.py)
#   vae32 · vae64 : SD 2.1 VAE latent(프레임 0, 4×64×64, v9_vae.py)를 train으로 맞춘 PCA 32 · 64차원
#   color : 물체 평균 Lab 3 + a*b* histogram 16 (v9_feat.py)
#   tex   : 방향 4 × 대역 3 log 에너지 12 (v9_feat.py)
#   v5    : 참고 = V-5 특징 9개(면적 제외: L_mean · L_std · grad · band 3 · chroma · a · b), 면적 회귀는 하지 않음
#  입력: enc(최종 Ours 7명 모델 고정 인코더 출력) · v1v4 · cortex (v5_read.py와 같음, 4번은 인코더 피험자 자리 0)
#  잴 것 (train 1,300 → test 104, λ는 차원마다 train 5겹):
#   r_mean : 차원별 test 상관의 평균 (train 표준편차가 0인 차원 제외), p = test 물체 순서를 섞은 순열 검정(10,000번, 한쪽)
#   ident  : 같은 범주 test 물체 중 예측 벡터와 상관이 가장 큰 정답 벡터가 자기 것인 비율 (무작위 약 0.125), p = 범주 안에서 정답 배정을 섞은 순열 검정
#  출력: <out>/read.json, <out>/pred_<피험자>.npz (enc 입력의 test 예측 — 범주 뺀 값, 원 단위)
# 사용 (실행 위치 $W/run): python v9_read.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_full/ausvp/model_final.pt --v9 v9_feat.npz --v5 v5_feat.npz --vae v9_vae.npz --subs … --out …
import os, sys, json, time, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
for k in ('repo', 'data_root', 'ckpt', 'v9', 'v5', 'subs', 'out'): p.add_argument('--' + k, required=True)
p.add_argument('--vae', default=''); p.add_argument('--v9b', default=''); p.add_argument('--targets', default=''); p.add_argument('--nperm', type=int, default=10000); p.add_argument('--inputs', default='enc,v1v4,cortex')
a = p.parse_args(); dev = 'cuda'; t0 = time.time(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval(); ENC = model.encode_frames
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
cat = lambda k: k.split('/')[0]
Z9 = np.load(a.v9); Z5 = np.load(a.v5); K9 = {str(k): i for i, k in enumerate(Z9['keys'])}; K5 = {str(k): i for i, k in enumerate(Z5['keys'])}
keys_ok = set(K9) & set(K5)
if a.vae: ZV = np.load(a.vae); KV = {str(k): i for i, k in enumerate(ZV['keys'])}; keys_ok &= set(KV)
TR = [k for k in lst('train') if k in keys_ok]; TE = [k for k in lst('test') if k in keys_ok]; CL = sorted({cat(k) for k in TR})
def g9(n): return lambda ks: Z9[n][[K9[k] for k in ks]].reshape(len(ks), -1).astype(np.float64)
TG = {'blur8_f0': g9('blur8_f0'), 'blur16_f0': g9('blur16_f0'), 'blur8_avg': g9('blur8_avg'), 'blur16_avg': g9('blur16_avg'),
      'color': lambda ks: np.concatenate([g9('lab_mean')(ks), g9('ab_hist')(ks)], 1), 'tex': g9('tex_ob')}
if 'mask8_avg' in Z9.files:   # 실루엣 대조 (11-28): 마스크만 줄인 것 + 흐릿한 RGB에서 같은 칸 마스크의 선형 몫(train으로 맞춤)을 뺀 나머지
    for n in ('mask8_f0', 'mask8_avg', 'mask16_avg'): TG[n] = g9(n)
    def _nomask(ks, _c={}):
        X = Z9['blur8_avg'][[K9[k] for k in ks]].astype(np.float64); M = Z9['mask8_avg'][[K9[k] for k in ks]].astype(np.float64)[..., None]
        if not _c:
            Xt = Z9['blur8_avg'][[K9[k] for k in TR]].astype(np.float64); Mt = Z9['mask8_avg'][[K9[k] for k in TR]].astype(np.float64)[..., None]
            mx, mm = Xt.mean(0), Mt.mean(0); _c['b'] = ((Xt - mx) * (Mt - mm)).sum(0) / (((Mt - mm) ** 2).sum(0) + 1e-9); _c['mx'], _c['mm'] = mx, mm
        return (X - _c['mx'] - _c['b'] * (M - _c['mm'])).reshape(len(ks), -1)
    TG['blur8_avg_nomask'] = _nomask
def _resid(X, M, tr_rows):   # 칸마다 X ~ β · M + c (train으로 맞춤) 의 나머지 — X [n, …, C], M [n, …, 1]
    mx, mm = X[tr_rows].mean(0), M[tr_rows].mean(0); b = ((X[tr_rows] - mx) * (M[tr_rows] - mm)).sum(0) / (((M[tr_rows] - mm) ** 2).sum(0) + 1e-9)
    return X - mx - b * (M - mm)
if a.v9b:   # 11-28 총괄 추가 ② 자르고 키운 흐릿한 이미지 ③ 고주파 에너지 공간 지도 (v9_feat2.py)
    ZB = np.load(a.v9b); KB = {str(k): i for i, k in enumerate(ZB['keys'])}; keys_ok &= set(KB)
    TR = [k for k in TR if k in keys_ok]; TE = [k for k in TE if k in keys_ok]
    _ALL = TR + TE; _IX = {k: i for i, k in enumerate(_ALL)}; _tr = np.arange(len(TR))
    def _arr(z, kk, n): return z[n][[kk[k] for k in _ALL]].astype(np.float64)
    _c = _arr(ZB, KB, 'crop8_avg'); _cm = _arr(ZB, KB, 'cmask8_avg')[..., None]
    _h4 = _arr(ZB, KB, 'hbmap4_avg'); _m4 = _arr(ZB, KB, 'mask4_avg'); _h8 = _arr(ZB, KB, 'hbmap8_avg'); _m8 = _arr(Z9, K9, 'mask8_avg')
    D2 = {'crop8_avg': _c, 'cmask8_avg': _cm[..., 0], 'crop8_avg_nomask': _resid(_c, _cm, _tr), 'hbmap4_avg': _h4, 'hbmap4_nomask': _resid(_h4, _m4, _tr),
          'hbmap8_avg': _h8, 'hbmap8_nomask': _resid(_h8, _m8, _tr)}
    for n, X in D2.items(): TG[n] = (lambda X: lambda ks: X[[_IX[k] for k in ks]].reshape(len(ks), -1))(X)
    # ① 질감 통계 12개를 train(범주 평균 뺀 값, train 표준편차로 나눔)의 PCA로 돌림 → 성분마다 따로 읽음
    _t = Z9['tex_ob'].astype(np.float64); _ttr = _t[[K9[k] for k in TR]]
    _cmt = {c: _ttr[[i for i, k in enumerate(TR) if cat(k) == c]].mean(0) for c in CL}
    _rt = _ttr - np.stack([_cmt[cat(k)] for k in TR]); _sdt = _rt.std(0); _, _St, _Vt = np.linalg.svd(_rt / _sdt, full_matrices=False)
    print('질감 PCA 설명 분산 비율', np.round(_St ** 2 / (_St ** 2).sum(), 3).tolist(), flush=True)
    print('질감 PC1 ~ 3 적재 (대역 저 · 중 · 고 × 방향 0 · 45 · 90 · 135):', [np.round(_Vt[i], 2).tolist() for i in range(3)], flush=True)
    for i in range(6): TG[f'tex_pc{i + 1}'] = (lambda i: lambda ks: (Z9['tex_ob'][[K9[k] for k in ks]].astype(np.float64) / _sdt) @ _Vt[i][:, None])(i)
N5 = [str(x) for x in Z5['names']]; I5 = [N5.index(n) for n in N5 if n != 'area']
TG['v5'] = lambda ks: Z5['feat'][[K5[k] for k in ks]][:, I5].astype(np.float64)
if a.vae:
    Lt = ZV['lat'][[KV[k] for k in TR]].reshape(len(TR), -1).astype(np.float64); mu_v = Lt.mean(0)
    U, S, Vt = np.linalg.svd(Lt - mu_v, full_matrices=False)                       # PCA를 train으로만 맞춤
    for d in (32, 64): TG[f'vae{d}'] = (lambda d: lambda ks: (ZV['lat'][[KV[k] for k in ks]].reshape(len(ks), -1).astype(np.float64) - mu_v) @ Vt[:d].T)(d)
    print('VAE PCA 설명 분산 32 · 64:', round(float((S[:32] ** 2).sum() / (S ** 2).sum()), 3), round(float((S[:64] ** 2).sum() / (S ** 2).sum()), 3), flush=True)
if a.targets: TG = {n: f for n, f in TG.items() if n in a.targets.split(',')}
Y = {}
for n, f in TG.items():
    ytr, yte = f(TR), f(TE); cm = {c: ytr[[i for i, k in enumerate(TR) if cat(k) == c]].mean(0) for c in CL}
    rtr = ytr - np.stack([cm[cat(k)] for k in TR]); rte = yte - np.stack([cm[cat(k)] for k in TE]); sd = rtr.std(0); ok = sd > 1e-6
    allr = np.concatenate([rtr, rte]); allv = np.concatenate([ytr, yte])
    Y[n] = dict(rtr=rtr[:, ok], rte=rte[:, ok], sd=sd[ok], ok=ok, wratio=float(allr[:, ok].var(0).sum() / allv[:, ok].var(0).sum()))
    print(f'정답 {n}: 차원 {ok.sum()} / {len(ok)} · 범주 안 분산 비율 {Y[n]["wratio"]:.3f}', flush=True)
print(f'train {len(TR)} · test {len(TE)} · 범주 {len(CL)}', flush=True)

lab = torch.as_tensor(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mmp_256_regions.npy'))).long().flatten().to(dev)
NAME = {int(k): v for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
MV = torch.isin(lab, torch.tensor([r for r, n in NAME.items() if n.split('_', 1)[-1] in {'V1', 'V2', 'V3', 'V4'}], device=dev)); MC = lab > 0
def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m
def raw(s, keys): return torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in keys]))
INP = a.inputs.split(',')
@torch.no_grad()
def feats(s, keys, si):
    o = {k: [] for k in INP}
    for i in range(0, len(keys), 32):
        x = raw(s, keys[i:i + 32]).to(dev)
        if 'enc' in o:
            with torch.autocast('cuda', dtype=torch.bfloat16): o['enc'].append(ENC(x, torch.full((len(x),), si, dtype=torch.long, device=dev)).float().mean(1).flatten(1).half())
        r = norm_in(x).mean(1).flatten(1)
        if 'v1v4' in o: o['v1v4'].append(r[:, MV].half())
        if 'cortex' in o: o['cortex'].append(r[:, MC].half())
    return {k: torch.cat(v) for k, v in o.items()}
def solve(K, T, lam): return torch.linalg.solve(K.double() + lam * K.diagonal().mean().double() * torch.eye(len(K), device=dev, dtype=torch.float64), T.double())
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
def colcorr(P, T):
    P = P - P.mean(0); T = T - T.mean(0); return (P * T).sum(0) / (np.sqrt((P ** 2).sum(0) * (T ** 2).sum(0)) + 1e-12)
def fit(Ktr, Kte, T):
    n = len(Ktr); fold = torch.arange(n, device=dev) % 5; Tt = torch.as_tensor(T, device=dev); NF = T.shape[1]; cv = np.zeros((len(LAMS), NF))
    for li, lam in enumerate(LAMS):
        P = torch.zeros_like(Tt, dtype=torch.float64)
        for f in range(5):
            a_, b_ = fold != f, fold == f; P[b_] = Ktr[b_][:, a_].double() @ solve(Ktr[a_][:, a_], Tt[a_], lam)
        cv[li] = colcorr(P.cpu().numpy(), T)
    best = cv.argmax(0); out = np.zeros((len(Kte), NF))
    for li in sorted(set(best)):
        W = solve(Ktr, Tt, LAMS[li]); P = (Kte.double() @ W).cpu().numpy(); out[:, best == li] = P[:, best == li]
    return out
def zs(X): return (X - X.mean(0)) / (X.std(0) + 1e-12)
CATS = np.array([cat(k) for k in TE]); GRP = [np.where(CATS == c)[0] for c in sorted(set(CATS))]
PERM = np.stack([rng.permutation(len(TE)) for _ in range(a.nperm)])
GPERM = [np.stack([rng.permutation(len(g)) for _ in range(a.nperm)]) for g in GRP]
def score(P, T):   # P, T: test [n, D] (train 표준편차로 나눈 값)
    r = colcorr(P, T); rm = float(r.mean())
    M = zs(P) @ zs(T).T / P.shape[1]                                     # M[i, j] = Σ_d zP[i,d] zT[j,d] / D → 대각합 / n = 차원별 상관 평균
    rp = M[np.arange(len(M))[None], PERM].mean(1); p_r = float(((rp >= rm - 1e-12).sum() + 1) / (a.nperm + 1))
    # 고르기: 벡터 상관 (물체마다 차원 방향으로 z)
    Pz = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-12); Tz = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-12)
    C = Pz @ Tz.T / P.shape[1]; hit = 0; null = np.zeros(a.nperm)
    for g, gp in zip(GRP, GPERM):
        am = C[np.ix_(g, g)].argmax(1); hit += int((am == np.arange(len(g))).sum()); null += (am[None] == gp).sum(1)
    idn = hit / len(TE); p_i = float(((null >= hit).sum() + 1) / (a.nperm + 1))
    return {'r_mean': round(rm, 4), 'p_r': round(p_r, 5), 'ident': round(idn, 4), 'p_ident': round(p_i, 5)}

CH = float(np.mean([1 / len(g) for g in GRP for _ in g]))
R = {'targets': {n: {'dims': int(v['ok'].sum()), 'within_ratio': round(v['wratio'], 4)} for n, v in Y.items()}, 'ident_chance': round(CH, 4), 'n_train': len(TR), 'n_test': len(TE), 'subs': {}}
print('고르기 무작위', round(CH, 4), flush=True)
for s in a.subs.split(','):
    si = SUBS.index(s) if s in SUBS else 0; ftr, fte = feats(s, TR, si), feats(s, TE, si); r = {}; sv = {'name': np.array(TE)}
    for kind in INP:
        xtr = ftr[kind].float(); mu = xtr.mean(0, keepdim=True); xtr = F.normalize(xtr - mu, dim=-1); xte = F.normalize(fte[kind].float() - mu, dim=-1)
        Ktr, Kte = xtr @ xtr.t(), xte @ xtr.t(); del xtr, xte; r[kind] = {}
        for n, v in Y.items():
            P = fit(Ktr, Kte, v['rtr'] / v['sd']); r[kind][n] = score(P, v['rte'] / v['sd'])
            if kind == 'enc': sv[f'pred_{n}'] = P * v['sd']; sv[f'true_{n}'] = v['rte']; sv[f'ok_{n}'] = v['ok']
        del Ktr, Kte
    R['subs'][s] = r; np.savez(f'{a.out}/pred_{s}.npz', **sv); del ftr, fte; torch.cuda.empty_cache()
    print(f'[{s}] {time.time() - t0:.0f}s | ' + ' | '.join(f'{k}: ' + ' '.join(f'{n} {v["r_mean"]:.3f}/{v["ident"]:.3f}' for n, v in r[k].items()) for k in r), flush=True)
    json.dump(R, open(f'{a.out}/read.json', 'w'), ensure_ascii=False, indent=1)
print('V9_READ_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
