# -*- coding: utf-8 -*-
# 11-19 진단: 프레임별 실루엣 차이를 비선형(MLP)으로 읽으면 선형(diag_vframe.py)보다 나아지나
#  정답 · 입력 · 지표는 diag_vframe.py와 같음 (프레임 차이 = 프레임 실루엣 − 물체 평균 − train 프레임별 평균, 입력도 같은 식으로 뺌)
#  입력을 train PCA 1024 성분으로 줄인 뒤: 선형(릿지, 같은 PCA 입력) vs MLP(1024 → 1024 → 1024 → 1024, GELU, dropout 0.5, AdamW, 300 epoch)
#  MLP는 hold 물체의 obj_top1이 가장 높은 epoch를 골라 test에서 잼. 시드 3개 평균
# 사용 (실행 위치 $W/run): python diag_vframe_mlp.py --data_root $DR --tgt $TGT --subs sub-0001,sub-0006 --out $W/prism_calib/vframe_mlp.json
import os, sys, json, time, argparse, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True); p.add_argument('--subs', required=True)
p.add_argument('--holdout', type=int, default=8); p.add_argument('--out', required=True); p.add_argument('--lags', default='0,1'); p.add_argument('--k', type=int, default=1024)
a = p.parse_args()
dev = 'cuda'; t0 = time.time(); SUBS = a.subs.split(',')
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
tr_all = lst('train'); by = {}
for k_ in tr_all: by.setdefault(k_.split('/')[0], []).append(k_)
H = {k_ for v in by.values() for k_ in v[-a.holdout:]}
SM = np.load(f'{a.tgt}/silhouettes32.npz'); SIL = {str(u): SM['sil'][i].astype(np.float32).reshape(10, -1) / 255.0 for i, u in enumerate(SM['uids'])}
uid = lambda k_: k_.split('/')[1]
has = lambda k_: uid(k_) in SIL and all(os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')) for s in SUBS)
SPL = {'train': [k_ for k_ in tr_all if k_ not in H and has(k_)], 'hold': [k_ for k_ in tr_all if k_ in H and has(k_)], 'test': [k_ for k_ in lst('test') if has(k_)]}
CL = sorted(by); Y = {s: torch.tensor([CL.index(k_.split('/')[0]) for k_ in v], device=dev) for s, v in SPL.items()}
T = {s: torch.tensor(np.stack([SIL[uid(k_)] for k_ in v]), device=dev) for s, v in SPL.items()}
here = os.path.dirname(os.path.abspath(__file__))
lab = torch.as_tensor(np.load(os.path.join(here, 'assets', 'mmp_256_regions.npy'))).long().to(dev)
NAME = {int(k): v for k, v in json.load(open(os.path.join(here, 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
EV = {'V1', 'V2', 'V3', 'V4'}; ALL = (lab > 0).flatten()
G = {'초기시각 V1-V4': torch.isin(lab, torch.tensor([r for r, n in NAME.items() if n.split('_', 1)[-1] in EV], device=dev)).flatten()[ALL], '전체': torch.ones(int(ALL.sum()), dtype=torch.bool, device=dev)}


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


@torch.no_grad()
def collect(split, s):
    o = []
    for i in range(0, len(SPL[split]), 32):
        x = norm_in(torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')).astype(np.float32)[:10] for k_ in SPL[split][i:i + 32]])).to(dev))
        o.append(x.flatten(2)[:, :, ALL].half())
    return torch.cat(o)


def pcorr(A, B): A = A - A.mean(1, keepdim=True); B = B - B.mean(1, keepdim=True); return float((F.normalize(A, dim=-1) * F.normalize(B, dim=-1)).sum(-1).mean())
D = {s: T[s] - T[s].mean(1, keepdim=True) for s in T}; muT = D['train'].mean(0, keepdim=True); Dc = {s: D[s] - muT for s in D}
R = {}
for s in SUBS:
    X = {sp: collect(sp, s) for sp in SPL}; print(f'  {s} 특징 끝 {time.time() - t0:.0f}s', flush=True)
    for lag in [int(v) for v in a.lags.split(',')]:
        FR = list(range(0, 10 - lag)); nf = len(FR)
        for g, gm in G.items():
            def feat(sp):
                x = X[sp][:, :, gm].float(); x = x - x.mean(1, keepdim=True); return x[:, [k + lag for k in FR]]
            xtr = feat('train'); mu = xtr.mean(0, keepdim=True); Fx = {sp: (feat(sp) - mu).flatten(0, 1) for sp in SPL}
            U, S_, V = torch.pca_lowrank(Fx['train'], q=a.k, center=False, niter=3)                    # train PCA
            Zx = {sp: (Fx[sp] @ V) / S_.mean() for sp in SPL}; Ty = {sp: Dc[sp][:, FR].flatten(0, 1) for sp in SPL}; del Fx
            def obj_top1(P, sp):
                Pn = F.normalize(P, dim=-1).view(-1, nf, 1024); Tn = F.normalize(Ty[sp], dim=-1).view(-1, nf, 1024); y = Y[sp]; hit = []
                for f_ in range(nf):
                    Sm = Pn[:, f_] @ Tn[:, f_].t()
                    for i in range(len(y)):
                        m = (y == y[i]).nonzero().squeeze(1); hit.append(float(m[Sm[i, m].argmax()] == i))
                return float(np.mean(hit))
            def frame_top1(P, sp):
                Pn = F.normalize(P, dim=-1).view(-1, nf, 1024); Tn = F.normalize(Ty[sp], dim=-1).view(-1, nf, 1024)
                return float((torch.einsum('nfd,ngd->nfg', Pn, Tn).argmax(-1) == torch.arange(nf, device=dev)).float().mean())
            # 선형 (같은 PCA 입력, 릿지 λ는 hold)
            def ridge(lam): A_ = torch.linalg.solve(Zx['train'].t() @ Zx['train'] + lam * torch.eye(a.k, device=dev) * (Zx['train'] ** 2).sum(0).mean(), Zx['train'].t() @ Ty['train']); return A_
            best = max((obj_top1(Zx['hold'] @ ridge(l), 'hold'), l) for l in (0.1, 1, 10, 100))[1]; Pl = Zx['test'] @ ridge(best)
            r = {'linear': {'obj_top1': round(obj_top1(Pl, 'test'), 4), 'frame_top1': round(frame_top1(Pl, 'test'), 4), 'corr': round(pcorr(Pl, Ty['test']), 4)}}
            # MLP (시드 3개)
            ms = []
            for seed in range(3):
                torch.manual_seed(seed)
                net = nn.Sequential(nn.Dropout(0.5), nn.Linear(a.k, 1024), nn.GELU(), nn.Dropout(0.5), nn.Linear(1024, 1024), nn.GELU(), nn.Dropout(0.5), nn.Linear(1024, 1024)).to(dev)
                opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.05); n = len(Zx['train']); bh, bt = -1, None
                for ep in range(300):
                    net.train(); perm = torch.randperm(n, device=dev)
                    for i in range(0, n, 512):
                        idx = perm[i:i + 512]; loss = F.mse_loss(net(Zx['train'][idx]), Ty['train'][idx]) * 100; opt.zero_grad(); loss.backward(); opt.step()
                    if ep % 10 == 9:
                        net.eval()
                        with torch.no_grad():
                            h = obj_top1(net(Zx['hold']), 'hold')
                            if h > bh: bh = h; Pt = net(Zx['test']); bt = (obj_top1(Pt, 'test'), frame_top1(Pt, 'test'), pcorr(Pt, Ty['test']), ep + 1)
                ms.append(bt)
            r['mlp'] = {'obj_top1': round(float(np.mean([m[0] for m in ms])), 4), 'frame_top1': round(float(np.mean([m[1] for m in ms])), 4), 'corr': round(float(np.mean([m[2] for m in ms])), 4),
                        'obj_top1_seeds': [round(m[0], 4) for m in ms], 'epochs': [m[3] for m in ms]}
            r['frame_chance'] = round(1 / nf, 4)
            R[f'{s} lag{lag} {g}'] = r; print(f'[{s} lag{lag} {g}]', json.dumps(r, ensure_ascii=False), flush=True)
            del Zx, V; torch.cuda.empty_cache()
    del X; torch.cuda.empty_cache()
R['obj_chance'] = round(float(np.mean([1 / int((Y['test'] == c).sum()) for c in Y['test']])), 4)
json.dump(R, open(a.out, 'w'), ensure_ascii=False, indent=1); print('VFRAME_MLP_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
