# -*- coding: utf-8 -*-
# 11-24 V-5: 자극의 저수준 외형(밝기 · 대비 · 질감 · 색)이 범주와 별개로 fMRI에서 읽히나 (학습 없음, 커널 릿지 — v3_read.py와 같은 방식)
#  특징: v5_feat.py 결과(<feat> npz, 물체마다 10개: L_mean · L_std · grad · band_lo · band_mid · band_hi · chroma · a_mean · b_mean · area, 영상 10프레임 평균)
#  입력 세 가지 (fMRI 프레임 2 ~ 7 평균, v3_read.py와 같음):
#    enc   = 최종 Ours 7명 모델의 고정 인코더 출력 토큰 (257 × 1024). 4번은 학습에 없어서 인코더 피험자 자리 0(=1번)으로 (v3_read.py와 같음)
#    v1v4  = 원 신호 평면 지도(영상 정규화 뒤) 중 HCP-MMP V1 · V2 · V3 · V4 픽셀
#    cortex= 원 신호 평면 지도 중 영역 라벨이 있는 피질 전체 픽셀
#  잴 것 (train 1,300 → test 104, λ는 train 안 5겹):
#   ① 범주 안 분산 비율 = var(특징 − 정답 범주 평균) / var(특징), 1,404개 전체
#   ② 범주 평균을 뺀 값의 상관: "특징 − 정답 범주 train 평균"을 맞춘 모델의 test 예측과 test 정답(같이 뺀 값)의 상관 + 순열 검정 p (10,000번, 한쪽)
#   ③ 면적을 뺀 값: 범주 평균을 뺀 특징에서 범주 평균을 뺀 면적의 선형 몫을 뺀 나머지(train으로 맞춘 회귀)를 같은 방식으로 읽음 (밝기 · 대비가 크기의 대리값인지)
#   ④ 같은 범주 8개 중 원래 물체 찾기: 면적을 뺀 외형 특징 9개(area 제외, train 표준편차로 나눔)의 예측 벡터와 정답 벡터 상관이 같은 범주 test 물체 중 가장 큰 것이 원래 물체인 비율 (무작위 0.125)
#  출력: <out>/read.json, <out>/pred_<피험자>.npz
# 사용 (실행 위치 $W/run): python v5_read.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_full/ausvp/model_final.pt --feat $W/prism_v5/v5_feat.npz --subs sub-0001,... --out $W/prism_v5
import os, sys, json, time, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
for k in ('repo', 'data_root', 'ckpt', 'feat', 'subs', 'out'): p.add_argument('--' + k, required=True)
p.add_argument('--nperm', type=int, default=10000)
a = p.parse_args(); dev = 'cuda'; t0 = time.time(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval(); ENC = model.encode_frames
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
cat = lambda k: k.split('/')[0]
Z = np.load(a.feat); FK = {k: i for i, k in enumerate(Z['keys'])}; FE = Z['feat'].astype(np.float64); NM = [str(x) for x in Z['names']]; IA = NM.index('area')
TR = [k for k in lst('train') if k in FK]; TE = [k for k in lst('test') if k in FK]; CL = sorted({cat(k) for k in TR})
Ytr = FE[[FK[k] for k in TR]]; Yte = FE[[FK[k] for k in TE]]
CM = {c: Ytr[[i for i, k in enumerate(TR) if cat(k) == c]].mean(0) for c in CL}
Ctr = np.stack([CM[cat(k)] for k in TR]); Cte = np.stack([CM[cat(k)] for k in TE])
Rtr, Rte = Ytr - Ctr, Yte - Cte                                                  # 범주 평균을 뺀 값
ALL = np.concatenate([Ytr, Yte]); ALLC = np.concatenate([Ctr, Cte])
wratio = (ALL - ALLC).var(0) / ALL.var(0)                                         # ① (범주 평균은 train 기준)
# ③ 면적 몫 빼기: 특징마다 Rtr[:, f] ~ β · Rtr[:, area] + c (train으로 맞춤) → 나머지
beta = np.array([np.polyfit(Rtr[:, IA], Rtr[:, f], 1) if f != IA else np.array([0., 0.]) for f in range(len(NM))])
Atr = Rtr - (Rtr[:, [IA]] * beta[:, 0] + beta[:, 1]); Ate = Rte - (Rte[:, [IA]] * beta[:, 0] + beta[:, 1]); Atr[:, IA] = Rtr[:, IA]; Ate[:, IA] = Rte[:, IA]
print(f'train {len(TR)} · test {len(TE)} · 범주 {len(CL)} | 특징 {NM}', flush=True)
print('범주 안 분산 비율', dict(zip(NM, np.round(wratio, 3))), flush=True)

lab = torch.as_tensor(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mmp_256_regions.npy'))).long().flatten().to(dev)
NAME = {int(k): v for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
MV = torch.isin(lab, torch.tensor([r for r, n in NAME.items() if n.split('_', 1)[-1] in {'V1', 'V2', 'V3', 'V4'}], device=dev)); MC = lab > 0
print(f'픽셀 V1-V4 {int(MV.sum())} · 피질 전체 {int(MC.sum())}', flush=True)


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m
def raw(s, keys): return torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in keys]))
@torch.no_grad()
def feats(s, keys, si):
    o = {'enc': [], 'v1v4': [], 'cortex': []}
    for i in range(0, len(keys), 32):
        x = raw(s, keys[i:i + 32]).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16): o['enc'].append(ENC(x, torch.full((len(x),), si, dtype=torch.long, device=dev)).float().mean(1).flatten(1).half())
        r = norm_in(x).mean(1).flatten(1); o['v1v4'].append(r[:, MV].half()); o['cortex'].append(r[:, MC].half())
    return {k: torch.cat(v) for k, v in o.items()}
def solve(K, T, lam): return torch.linalg.solve(K.double() + lam * K.diagonal().mean().double() * torch.eye(len(K), device=dev, dtype=torch.float64), T.double())
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0); corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])
def fit(Ktr, Kte, T):                                                      # T [n, F] (train 표준화) → test 예측 [m, F], 특징별 λ (train 5겹 상관 최대)
    n = len(Ktr); fold = torch.arange(n, device=dev) % 5; T = torch.as_tensor(T, device=dev); NF = T.shape[1]; cv = np.zeros((len(LAMS), NF))
    for li, lam in enumerate(LAMS):
        P = torch.zeros_like(T, dtype=torch.float64)
        for f in range(5):
            a_, b_ = fold != f, fold == f; P[b_] = Ktr[b_][:, a_].double() @ solve(Ktr[a_][:, a_], T[a_], lam)
        Pn, Tn = P.cpu().numpy(), T.cpu().numpy(); cv[li] = [corr(Pn[:, d], Tn[:, d]) for d in range(NF)]
    best = cv.argmax(0); out = np.zeros((len(Kte), NF))
    for li in sorted(set(best)):
        W = solve(Ktr, T, LAMS[li]); P = (Kte.double() @ W).cpu().numpy(); out[:, best == li] = P[:, best == li]
    return out, [LAMS[i] for i in best], cv.max(0)
PI = None
def perm_p(x, y):                                                          # 한쪽 순열 검정 (상관이 클수록): y를 섞은 nperm개와 비교
    global PI
    if PI is None or PI.shape[1] != len(y): PI = np.stack([rng.permutation(len(y)) for _ in range(a.nperm)])
    zx = (x - x.mean()) / (x.std() + 1e-12); zy = (y - y.mean()) / (y.std() + 1e-12)
    r = float((zx * zy).mean()); rp = (zy[PI] * zx[None]).mean(1); return float(((rp >= r).sum() + 1) / (a.nperm + 1))
def ident(P, T, keys):                                                     # 같은 범주 test 물체 중 원래 물체 찾기 (상관 최대)
    hit = []
    for i, k in enumerate(keys):
        J = [j for j, k2 in enumerate(keys) if cat(k2) == cat(k)]
        sc = [corr(P[i], T[j]) for j in J]; hit.append(J[int(np.argmax(sc))] == i)
    return float(np.mean(hit)), float(np.mean([1 / sum(cat(k2) == cat(k) for k2 in keys) for k in keys]))

sd_r, sd_a = Rtr.std(0) + 1e-9, Atr.std(0) + 1e-9
R = {'features': NM, 'within_ratio': [round(float(x), 4) for x in wratio], 'n_train': len(TR), 'n_test': len(TE), 'area_beta': [round(float(b), 4) for b in beta[:, 0]], 'subs': {}}
IX = [i for i in range(len(NM)) if i != IA]
for s in a.subs.split(','):
    si = SUBS.index(s) if s in SUBS else 0; ftr, fte = feats(s, TR, si), feats(s, TE, si); r = {}; sv = {'name': np.array(TE), 'res_true': Rte, 'area_out_true': Ate}
    for kind in ('enc', 'v1v4', 'cortex'):
        xtr = ftr[kind].float(); mu = xtr.mean(0, keepdim=True); xtr = F.normalize(xtr - mu, dim=-1); xte = F.normalize(fte[kind].float() - mu, dim=-1)
        Ktr, Kte = xtr @ xtr.t(), xte @ xtr.t(); del xtr, xte
        Pr, lr, cvr = fit(Ktr, Kte, Rtr / sd_r); Pr = Pr * sd_r                 # ② 범주 평균을 뺀 값
        Pa, la, cva = fit(Ktr, Kte, Atr / sd_a); Pa = Pa * sd_a                 # ③ 면적 몫까지 뺀 값
        rr = [corr(Pr[:, f], Rte[:, f]) for f in range(len(NM))]; ra = [corr(Pa[:, f], Ate[:, f]) for f in range(len(NM))]
        pr_ = [perm_p(Pr[:, f], Rte[:, f]) for f in range(len(NM))]; pa_ = [perm_p(Pa[:, f], Ate[:, f]) for f in range(len(NM))]
        idn, ch = ident(Pa[:, IX] / sd_a[IX], Ate[:, IX] / sd_a[IX], TE)
        r[kind] = {'within_r': [round(x, 4) for x in rr], 'within_p': [round(x, 5) for x in pr_], 'area_out_r': [round(x, 4) for x in ra], 'area_out_p': [round(x, 5) for x in pa_],
                   'ident_top1': round(idn, 4), 'ident_chance': round(ch, 4), 'lam': lr, 'lam_area_out': la, 'cv_train': [round(float(x), 4) for x in cvr]}
        sv['pred_res_' + kind] = Pr; sv['pred_area_out_' + kind] = Pa; del Ktr, Kte
    R['subs'][s] = r; np.savez(f'{a.out}/pred_{s}.npz', **sv); del ftr, fte; torch.cuda.empty_cache()
    print(f'[{s}] ' + ' | '.join(f'{k}: 범주 뺀 상관 {dict(zip(NM, r[k]["within_r"]))} · 찾기 {r[k]["ident_top1"]}' for k in r) + f' | {time.time() - t0:.0f}s', flush=True)
    json.dump(R, open(f'{a.out}/read.json', 'w'), ensure_ascii=False, indent=1)
print('V5_READ_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
