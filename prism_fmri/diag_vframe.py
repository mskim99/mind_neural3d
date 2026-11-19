# -*- coding: utf-8 -*-
# 11-19 진단: "프레임별 차이"(Variational이 맡는다고 가정한 정보)가 fMRI에서 선형으로 읽히나 — 학습 없음, 커널 릿지
#  정답: 영상 프레임 k(= fMRI 프레임 k 시점, round(19.2k))의 실루엣 32×32 − 그 물체의 10프레임 평균 = 프레임 차이 (물체가 돌면서 생기는 변화)
#        거기서 다시 train 물체의 프레임별 평균을 뺌 (모든 물체에 공통인 시간 변화는 프레임 번호만으로 맞힐 수 있으므로 제외)
#  입력: fMRI 프레임 k + lag (lag = 0 · 1 · 2, 혈류 지연), in_norm, 영역 픽셀. 같은 식으로 물체 평균 · 프레임별 train 평균을 뺌
#  지표 (test 물체):
#    obj_top1   같은 프레임 · 같은 범주 test 물체 중 정답 물체 고르기 (우연 ≈ 1/8) — 프레임 차이가 물체마다 다르게 읽히나
#    frame_top1 같은 물체의 프레임들 중 정답 프레임 고르기 (우연 = 1/프레임 수) — 물체 고유의 프레임 순서가 읽히나
#    corr       예측 · 정답 프레임 차이 지도의 화소 상관
#  참고: 프레임 차이의 크기 = 프레임 차이 분산 / 물체 평균 지도의 물체 간 분산
# 사용 (실행 위치 $W/run): python diag_vframe.py --data_root $DR --tgt $TGT --subs sub-0001,sub-0006 --out $W/prism_calib/vframe.json
import os, sys, json, time, argparse, numpy as np, torch
import torch.nn.functional as F
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True); p.add_argument('--subs', required=True)
p.add_argument('--holdout', type=int, default=8); p.add_argument('--out', required=True)
a = p.parse_args()
dev = 'cuda'; t0 = time.time(); SUBS = a.subs.split(',')
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
tr_all = lst('train'); by = {}
for k_ in tr_all: by.setdefault(k_.split('/')[0], []).append(k_)
H = {k_ for v in by.values() for k_ in v[-a.holdout:]}
SM = np.load(f'{a.tgt}/silhouettes32.npz'); SIL = {str(u): SM['sil'][i].astype(np.float32).reshape(10, -1) / 255.0 for i, u in enumerate(SM['uids'])}   # [10, 1024]
uid = lambda k_: k_.split('/')[1]
has = lambda k_: uid(k_) in SIL and all(os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')) for s in SUBS)
SPL = {'train': [k_ for k_ in tr_all if k_ not in H and has(k_)], 'hold': [k_ for k_ in tr_all if k_ in H and has(k_)], 'test': [k_ for k_ in lst('test') if has(k_)]}
CL = sorted(by); Y = {s: torch.tensor([CL.index(k_.split('/')[0]) for k_ in v], device=dev) for s, v in SPL.items()}
T = {s: torch.tensor(np.stack([SIL[uid(k_)] for k_ in v]), device=dev) for s, v in SPL.items()}            # [N, 10, 1024]
print('물체 ' + ' · '.join(f'{s} {len(v)}' for s, v in SPL.items()), flush=True)
here = os.path.dirname(os.path.abspath(__file__))
lab = torch.as_tensor(np.load(os.path.join(here, 'assets', 'mmp_256_regions.npy'))).long().to(dev)
NAME = {int(k): v for k, v in json.load(open(os.path.join(here, 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
EV = {'V1', 'V2', 'V3', 'V4'}
gmask = {'초기시각 V1-V4': torch.isin(lab, torch.tensor([r for r, n in NAME.items() if n.split('_', 1)[-1] in EV], device=dev)).flatten(), '전체': (lab > 0).flatten()}


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


@torch.no_grad()
def collect(split, s):
    o = []
    for i in range(0, len(SPL[split]), 32):
        x = norm_in(torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')).astype(np.float32)[:10] for k_ in SPL[split][i:i + 32]])).to(dev))
        o.append(x.flatten(2)[:, :, gmask['전체']].half())
    return torch.cat(o)                                                    # [N, 10, P]


def krr(Xtr, Ttr, Xev, lam):
    Kt = Xtr @ Xtr.t(); A = torch.linalg.solve(Kt + lam * Kt.diagonal().mean() * torch.eye(len(Kt), device=dev), Ttr)
    return Xev @ Xtr.t() @ A


def pcorr(A, B): A = A - A.mean(1, keepdim=True); B = B - B.mean(1, keepdim=True); return float((F.normalize(A, dim=-1) * F.normalize(B, dim=-1)).sum(-1).mean())


LAMS = (0.03, 0.1, 0.3, 1, 3, 10)
# 정답: 프레임 차이 (물체 평균 뺌) → train 프레임별 평균 뺌
D = {s: T[s] - T[s].mean(1, keepdim=True) for s in T}; muT = D['train'].mean(0, keepdim=True); Dc = {s: D[s] - muT for s in D}
R = {'프레임 차이 분산 / 물체 평균 지도의 물체 간 분산': round(float(D['test'].var() / (T['test'].mean(1) - T['test'].mean(1).mean(0)).var()), 4),
     '공통 시간 변화를 뺀 뒤 남는 프레임 차이 분산 비율': round(float(Dc['test'].var() / D['test'].var()), 4)}
print(json.dumps(R, ensure_ascii=False), flush=True)
sub_g = {g: gmask[g][gmask['전체']] for g in gmask}                         # '전체' 픽셀 안에서의 영역 위치
for s in SUBS:
    X = {sp: collect(sp, s) for sp in SPL}; print(f'  {s} 특징 끝 {time.time() - t0:.0f}s', flush=True)
    for lag in (0, 1, 2):
        FR = list(range(0, 10 - lag))                                       # 정답 프레임 k ↔ fMRI 프레임 k + lag
        for g, gm in sub_g.items():
            def feat(sp, mu=None):
                x = X[sp][:, :, gm].float(); x = x - x.mean(1, keepdim=True); x = x[:, [k + lag for k in FR]]           # 물체 평균 빼고 쓸 프레임만
                return x
            xtr = feat('train'); mu = xtr.mean(0, keepdim=True)
            Fx = {sp: F.normalize((feat(sp) - mu).flatten(0, 1), dim=-1) for sp in SPL}                                # [N·nf, P]
            Ty = {sp: Dc[sp][:, FR].flatten(0, 1) for sp in SPL}
            nf = len(FR)
            def obj_top1(P, sp):
                Pn = F.normalize(P, dim=-1).view(-1, nf, 1024); Tn = F.normalize(Ty[sp], dim=-1).view(-1, nf, 1024); y = Y[sp]; hit = []
                for f_ in range(nf):
                    Sm = Pn[:, f_] @ Tn[:, f_].t()
                    for i in range(len(y)):
                        m = (y == y[i]).nonzero().squeeze(1); hit.append(float(m[Sm[i, m].argmax()] == i))
                return float(np.mean(hit))
            def frame_top1(P, sp):
                Pn = F.normalize(P, dim=-1).view(-1, nf, 1024); Tn = F.normalize(Ty[sp], dim=-1).view(-1, nf, 1024)
                Sm = torch.einsum('nfd,ngd->nfg', Pn, Tn); return float((Sm.argmax(-1) == torch.arange(nf, device=dev)).float().mean())
            lam = max((obj_top1(krr(Fx['train'], Ty['train'], Fx['hold'], l), 'hold'), l) for l in LAMS)[1]
            P = krr(Fx['train'], Ty['train'], Fx['test'], lam)
            r = {'obj_top1': round(obj_top1(P, 'test'), 4), 'frame_top1': round(frame_top1(P, 'test'), 4), 'frame_chance': round(1 / nf, 4),
                 'corr': round(pcorr(P, Ty['test']), 4), 'lam': lam}
            R[f'{s} lag{lag} {g}'] = r; print(f'[{s} lag{lag} {g}]', json.dumps(r, ensure_ascii=False), flush=True)
            del Fx, P; torch.cuda.empty_cache()
    del X; torch.cuda.empty_cache()
R['obj_chance(같은 범주 test)'] = round(float(np.mean([1 / int((Y['test'] == c).sum()) for c in Y['test']])), 4)
json.dump(R, open(a.out, 'w'), ensure_ascii=False, indent=1); print('VFRAME_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
