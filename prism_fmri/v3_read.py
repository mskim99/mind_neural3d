# -*- coding: utf-8 -*-
# 11-24 V-3 (가): 물체의 세 축 길이 비율(가로 x · 세로 y · 깊이 z)을 fMRI에서 선형으로 읽을 수 있나 — train 1,300개로 맞추고 test 104개에서 정답과의 상관 (학습 없음, 커널 릿지)
#  정답: 형태 정답 점유 격자(shape_targets/voxels, 64³, ShapeNet 정규화 좌표 = 평가 정답 점구름 gt_pc와 같은 축 — test 104개에서 축별 상관 0.999 확인)의 축별 길이 e
#        → 크기를 뺀 비율 t = log e − mean(log e) (세 값, 합 0)
#  입력 두 가지: enc = 모델의 고정 인코더 출력 토큰(프레임 6개 평균, 257 × 1024) / raw = 원 신호(영상 정규화 뒤 평면 지도의 0 아닌 픽셀, 프레임 평균)
#  릿지 세기 λ는 train 안 5겹으로 고름 (test 안 씀). 모델 둘: 전체 값을 맞추는 것 / 범주 평균을 뺀 값을 맞추는 것
#  출력: <out>/read.json (피험자 · 입력 · 축별 상관: 전체 / 범주 평균을 뺀 값 / 범주 평균만 쓴 기준) + <out>/pred_<피험자>.npz (test 예측 · 정답)
# 사용 (실행 위치 $W/run): python v3_read.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_full/ausvp/model_final.pt --vox $W/data/shape_targets/voxels --subs sub-0001,... --out $W/prism_v3
import os, sys, json, time, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
for k in ('repo', 'data_root', 'ckpt', 'vox', 'subs', 'out'): p.add_argument('--' + k, required=True)
a = p.parse_args(); dev = 'cuda'; t0 = time.time(); os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval(); ENC = model.encode_frames
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
cat = lambda k: k.split('/')[0]; uid = lambda k: k.split('/')[1]


def ratio(u):                                                              # 점유 격자 → 축별 길이 → 로그 비율 [3]
    v = np.unpackbits(np.load(f'{a.vox}/{u}.npy')).reshape(64, 64, 64).astype(bool); e = []
    for d in range(3):
        ix = np.nonzero(v.any(tuple(x for x in range(3) if x != d)))[0]; e.append((ix.max() - ix.min() + 1) / 64)
    l = np.log(np.array(e)); return l - l.mean()


TR = [k for k in lst('train') if os.path.exists(f'{a.vox}/{uid(k)}.npy')]; TE = lst('test')
Ytr = np.stack([ratio(uid(k)) for k in TR]); Yte = np.stack([ratio(uid(k)) for k in TE]); CL = sorted({cat(k) for k in TR})
CM = {c: Ytr[[i for i, k in enumerate(TR) if cat(k) == c]].mean(0) for c in CL}
Ctr = np.stack([CM[cat(k)] for k in TR]); Cte = np.stack([CM[cat(k)] for k in TE])   # 정답 범주의 train 평균 비율
print(f'train {len(TR)} · test {len(TE)} · 범주 {len(CL)} | 정답 비율 표준편차(축별) {Yte.std(0).round(3)} · 범주 평균을 뺀 뒤 {(Yte - Cte).std(0).round(3)}', flush=True)


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m
def raw(s, keys): return torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in keys]))
M = (raw(a.subs.split(',')[0], TE[:4]) != 0).any(0).any(0).flatten().to(dev)
@torch.no_grad()
def feats(s, keys, si):                                                    # → {'enc': [n, 257·1024], 'raw': [n, P]} (프레임 평균)
    o = {'enc': [], 'raw': []}
    for i in range(0, len(keys), 32):
        x = raw(s, keys[i:i + 32]).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16): o['enc'].append(ENC(x, torch.full((len(x),), si, dtype=torch.long, device=dev)).float().mean(1).flatten(1).half())
        o['raw'].append(norm_in(x).mean(1).flatten(1)[:, M].half())
    return {k: torch.cat(v) for k, v in o.items()}
def solve(K, T, lam): return torch.linalg.solve(K.double() + lam * K.diagonal().mean().double() * torch.eye(len(K), device=dev, dtype=torch.float64), T.double())
LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0); corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])
def fit(Ktr, Kte, T):                                                      # T [n, 3] (평균 뺀 값) → test 예측 [m, 3], 고른 λ, train 5겹 상관
    n = len(Ktr); fold = torch.arange(n, device=dev) % 5; T = torch.as_tensor(T, device=dev); best = None
    for lam in LAMS:
        P = torch.zeros_like(T, dtype=torch.float64)
        for f in range(5):
            a_, b_ = fold != f, fold == f; P[b_] = Ktr[b_][:, a_].double() @ solve(Ktr[a_][:, a_], T[a_], lam)
        c = float(np.mean([corr(P[:, d].cpu().numpy(), T[:, d].cpu().numpy()) for d in range(3)]))
        if best is None or c > best[0]: best = (c, lam)
    return (Kte.double() @ solve(Ktr, T, best[1])).cpu().numpy(), best[1], round(best[0], 4)


R = {'axes': ['x 가로', 'y 세로', 'z 깊이'], 'n_train': len(TR), 'cat_only': {'total': [round(corr(Cte[:, d], Yte[:, d]), 4) for d in range(3)]}, 'subs': {}}
mu_y = Ytr.mean(0)
for s in a.subs.split(','):
    si = SUBS.index(s) if s in SUBS else 0; ftr, fte = feats(s, TR, si), feats(s, TE, si); r = {}; sv = {'name': np.array(TE), 'gt': Yte, 'cat_mean_true': Cte}
    for kind in ('enc', 'raw'):
        xtr = ftr[kind].float(); mu = xtr.mean(0, keepdim=True); xtr = F.normalize(xtr - mu, dim=-1); xte = F.normalize(fte[kind].float() - mu, dim=-1)
        Ktr, Kte = xtr @ xtr.t(), xte @ xtr.t(); del xtr, xte
        Pt, lt, ct = fit(Ktr, Kte, Ytr - mu_y); Pt = Pt + mu_y                    # 전체 값을 맞춘 모델
        Pr, lr, cr = fit(Ktr, Kte, Ytr - Ctr)                                      # 범주 평균을 뺀 값을 맞춘 모델
        r[kind] = {'total': [round(corr(Pt[:, d], Yte[:, d]), 4) for d in range(3)], 'within_cat': [round(corr(Pt[:, d] - Cte[:, d], Yte[:, d] - Cte[:, d]), 4) for d in range(3)],
                   'resfit_within_cat': [round(corr(Pr[:, d], Yte[:, d] - Cte[:, d]), 4) for d in range(3)], 'lam': lt, 'lam_res': lr, 'cv_train': ct, 'cv_train_res': cr,
                   'mae': round(float(np.abs(Pt - Yte).mean()), 4), 'mae_cat_true': round(float(np.abs(Cte - Yte).mean()), 4)}
        sv['pred_' + kind] = Pt; sv['predres_' + kind] = Pr; del Ktr, Kte
    R['subs'][s] = r; np.savez(f'{a.out}/pred_{s}.npz', **sv); del ftr, fte; torch.cuda.empty_cache()
    print(f'[{s}] ' + ' | '.join(f'{k}: 전체 {r[k]["total"]} · 범주 평균 뺀 값 {r[k]["within_cat"]} · (뺀 값을 맞춘 모델) {r[k]["resfit_within_cat"]} · λ {r[k]["lam"]}' for k in r) + f' | {time.time() - t0:.0f}s', flush=True)
    json.dump(R, open(f'{a.out}/read.json', 'w'), ensure_ascii=False, indent=1)
np.savez(f'{a.out}/cat_mean.npz', classes=np.array(CL), mean=np.stack([CM[c] for c in CL]))
print('범주 평균만 (정답 범주) 전체 상관', R['cat_only']['total'], flush=True); print('V3_READ_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
