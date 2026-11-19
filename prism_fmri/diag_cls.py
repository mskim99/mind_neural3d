# -*- coding: utf-8 -*-
# 11-19 진단: 범주(13개)가 fMRI에서 선형으로 얼마나 읽히나 — 학습 없음, 커널 릿지(one-hot) 분류
#  모델 범주 정확도(test 피험자 1 · 6 = 0.462 · 0.327)가 어디서 막히는지: 입력 영역 · 프레임 · 피험자 합치기에 따라 천장이 얼마나 바뀌나
#  입력 (in_norm 같은 식):
#    raw:<영역>     원 fMRI 평면 지도 그 영역 픽셀 (프레임 2~7 평균, 모델 입력과 같은 프레임)
#    raw10:전체     프레임 0~9 평균 (모델이 안 쓰는 프레임 포함)
#    rawcat:전체    프레임 2~7을 평균 내지 않고 이어 붙임 (시간 정보 유지)
#    vit            고정 UKB ViT 특징 (LayerNorm → 2×2 묶음 64개 → 프레임 평균 → 폄, diag_calib.py와 같음)
#    vit+raw        둘 이어 붙임 (각각 L2 정규화)
#  학습 데이터: own = 그 피험자만 / pool = 피험자 여러 명 데이터를 합쳐 하나의 분류기 (평면 지도는 피험자 공통 좌표) → test는 피험자마다
#  λ는 hold(범주마다 train 목록 끝 8개)로 고르고 test 104개에서 top-1 (우연 1/13 = 0.077)
# 사용 (실행 위치 $W/run): python diag_cls.py --repo $W/MinD-3D --data_root $DR --subs sub-0001,sub-0006 --pool sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008 --enc_path $U --out $W/prism_calib/cls_ceil.json
import os, sys, json, time, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import load_encoder

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True)
p.add_argument('--subs', required=True); p.add_argument('--pool', default=''); p.add_argument('--enc_path', required=True)
p.add_argument('--holdout', type=int, default=8); p.add_argument('--out', required=True)
a = p.parse_args()
dev = 'cuda'; t0 = time.time(); SUBS = a.subs.split(','); POOL = a.pool.split(',') if a.pool else []
ALLS = list(dict.fromkeys(SUBS + POOL))
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
tr_all = lst('train'); by = {}
for k_ in tr_all: by.setdefault(k_.split('/')[0], []).append(k_)
H = {k_ for v in by.values() for k_ in v[-a.holdout:]}
has = lambda k_: all(os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')) for s in ALLS)
SPL = {'train': [k_ for k_ in tr_all if k_ not in H and has(k_)], 'hold': [k_ for k_ in tr_all if k_ in H and has(k_)], 'test': [k_ for k_ in lst('test') if has(k_)]}
CL = sorted(by); Y = {s: torch.tensor([CL.index(k_.split('/')[0]) for k_ in v], device=dev) for s, v in SPL.items()}
print('물체 ' + ' · '.join(f'{s} {len(v)}' for s, v in SPL.items()) + f' | 범주 {len(CL)} | 피험자 {ALLS}', flush=True)

here = os.path.dirname(os.path.abspath(__file__))
lab = torch.as_tensor(np.load(os.path.join(here, 'assets', 'mmp_256_regions.npy'))).long().to(dev)
NAME = {int(k): v for k, v in json.load(open(os.path.join(here, 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
GROUP = {'초기시각 V1-V4': {'V1', 'V2', 'V3', 'V4'},
         '등쪽': {'V3A', 'V3B', 'V6', 'V6A', 'V7', 'IPS1', 'MT', 'MST', 'LO3'},
         '배쪽·LO': {'LO1', 'LO2', 'PIT', 'FFC', 'V8', 'VVC', 'PH', 'VMV1', 'VMV2', 'VMV3', 'TE2p', 'TF'}}
gmask = {g: torch.isin(lab, torch.tensor([r for r, n in NAME.items() if n.split('_', 1)[-1] in names], device=dev)).flatten() for g, names in GROUP.items()}
gmask['전체'] = (lab > 0).flatten()
for g, m in gmask.items(): print(f'  영역 {g}: 픽셀 {int(m.sum())}', flush=True)

from omegaconf import OmegaConf
enc = load_encoder(a.enc_path, OmegaConf.load(os.path.join(a.repo, 'configs', 'mind3d_pp.yaml'))).to(dev).eval().requires_grad_(False)


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


@torch.no_grad()
def collect(split, s):
    o = {k: [] for k in ('r', 'r10', 'rc', 'v')}
    for i in range(0, len(SPL[split]), 16):
        x10 = norm_in(torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')).astype(np.float32)[:10] for k_ in SPL[split][i:i + 16]])).to(dev))
        x = x10[:, 2:8]; B, T = x.shape[:2]
        o['r'].append(x.mean(1).flatten(1).half()); o['r10'].append(x10.mean(1).flatten(1).half())
        o['rc'].append(x.flatten(2)[:, :, gmask['전체']].flatten(1).half())
        with torch.autocast('cuda', dtype=torch.bfloat16):
            t = enc.forward_encoder_wo_pred(x.reshape(B * T, 1, 256, 256)).view(B, T, 257, -1)[:, :, 1:].float()
        o['v'].append(F.layer_norm(t, t.shape[-1:]).view(B, T, 8, 2, 8, 2, -1).mean((3, 5)).flatten(2).mean(1).half())
    return {k: torch.cat(v) for k, v in o.items()}


def feats(D):                                   # D[split] = collect 결과 → {이름: {split: [N, d] 정규화}}
    nz = lambda t: F.normalize(t.float(), dim=-1)
    out = {f'raw:{g}': {sp: nz(D[sp]['r'][:, m]) for sp in D} for g, m in gmask.items()}
    out['raw10:전체'] = {sp: nz(D[sp]['r10'][:, gmask['전체']]) for sp in D}
    out['rawcat:전체'] = {sp: nz(D[sp]['rc']) for sp in D}
    out['vit'] = {sp: nz(D[sp]['v']) for sp in D}
    out['vit+raw'] = {sp: torch.cat([out['vit'][sp], out['raw:전체'][sp]], 1) / 2 ** 0.5 for sp in D}
    return out


def center(Xs):                                 # train 평균 빼기
    mu = Xs['train'].mean(0); return {s: F.normalize(v - mu, dim=-1) for s, v in Xs.items()}


def krr(Xtr, Ttr, Xev, lam):
    Kt = Xtr @ Xtr.t(); A = torch.linalg.solve(Kt + lam * Kt.diagonal().mean() * torch.eye(len(Kt), device=dev), Ttr)
    return Xev @ Xtr.t() @ A


LAMS = (0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)
acc = lambda P, y: float((P.argmax(-1) == y).float().mean())
R = {'chance': round(1 / len(CL), 4), 'n': {s: len(v) for s, v in SPL.items()}}
FE = {}
for s in ALLS:
    D = {sp: collect(sp, s) for sp in SPL}; FE[s] = {k: center(v) for k, v in feats(D).items()}; del D
    print(f'  {s} 특징 끝 {time.time() - t0:.0f}s', flush=True)
OH = {sp: F.one_hot(Y[sp], len(CL)).float() for sp in SPL}
for fn in FE[SUBS[0]]:
    for s in SUBS:                               # own: 그 피험자만
        X = FE[s][fn]; lam = max((acc(krr(X['train'], OH['train'], X['hold'], l), Y['hold']), l) for l in LAMS)[1]
        r = {'own': round(acc(krr(X['train'], OH['train'], X['test'], lam), Y['test']), 4), 'lam_own': lam}
        for pn, PS in (('pool2', SUBS), ('pool7', POOL)):
            if not PS or len(PS) < 2 or fn == 'rawcat:전체': continue      # rawcat은 차원이 커서(28만) 합치기 생략
            Xtr = torch.cat([FE[q][fn]['train'] for q in PS]); Ttr = OH['train'].repeat(len(PS), 1)
            Xh = torch.cat([FE[q][fn]['hold'] for q in PS]); Yh = Y['hold'].repeat(len(PS))
            lam = max((acc(krr(Xtr, Ttr, Xh, l), Yh), l) for l in LAMS)[1]
            r[pn] = round(acc(krr(Xtr, Ttr, X['test'], lam), Y['test']), 4); r['lam_' + pn] = lam
        R[f'{s} {fn}'] = r; print(f'[{s} {fn}]', json.dumps(r, ensure_ascii=False), flush=True)
json.dump(R, open(a.out, 'w'), ensure_ascii=False, indent=1); print('CLS_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
