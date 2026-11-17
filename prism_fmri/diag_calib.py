# -*- coding: utf-8 -*-
# 11-18 진단: 1단계 보정(calib.pt)이 고정 ViT 특징에서 범주 · 물체 정보를 지웠는지 (학습 없음, 선형 읽기만)
#  특징 = train_calib.py feats()와 같음 (in_norm → 보정 → 고정 ViT → patch 프레임 평균 → LayerNorm → 2×2 묶음 64개 → 폄), 보정 끔 · 켬 두 가지
#  재는 것 (피험자마다, 자기 train 평균 빼고 정규화 = train_calib center와 같음):
#   ① 범주 선형 읽기: 커널 릿지(선형 커널), train으로 맞추고 test 범주 정확도. λ는 hold로 고름. 같은 피험자 / 다른 피험자로 옮겨 읽기(1 → 6, 6 → 1)
#   ② 같은 피험자 안 물체 구별: test에서 프레임 앞 3개 · 뒤 3개 평균 특징끼리 같은 물체 고르기 top-1
#   ③ 다른 피험자 같은 물체 고르기 (train_calib evaluate와 같음), test
#   ④ 보정이 바꾼 크기: 입력 ‖x' − x‖ / ‖x‖, 특징 cos(보정 전, 보정 후), 물체 간 분산 비율(중심 뺀 특징 노름 ÷ 원 특징 노름)
# 사용 (실행 위치 $W/run): python diag_calib.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs sub-0001,sub-0006 --enc_path $U --calib $W/prism_calib/s16/calib.pt --out $W/prism_calib/s16/diag.json
import os, sys, json, time, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import RegionCalib, load_encoder

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--subs', required=True); p.add_argument('--enc_path', required=True); p.add_argument('--calib', required=True)
p.add_argument('--in_norm', type=int, default=1); p.add_argument('--holdout', type=int, default=8); p.add_argument('--out', required=True)
p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
dev = 'cuda'; FRAMES = list(range(2, 8)); SUBS = a.subs.split(','); K = len(SUBS)
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
tr_all = lst('train'); by = {}
for k_ in tr_all: by.setdefault(k_.split('/')[0], []).append(k_)
H = {k_ for v in by.values() for k_ in v[-a.holdout:]}
has = lambda k_: os.path.exists(f'{a.tgt}/dino/{k_.split("/")[1]}.npy') and all(os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')) for s in SUBS)
SPL = {'train': [k_ for k_ in tr_all if k_ not in H and has(k_)], 'hold': [k_ for k_ in tr_all if k_ in H and has(k_)], 'test': [k_ for k_ in lst('test') if has(k_)]}
if a.limit: SPL = {s: v[:a.limit] for s, v in SPL.items()}
CL = sorted({k_.split('/')[0] for v in SPL.values() for k_ in v}); Y = {s: torch.tensor([CL.index(k_.split('/')[0]) for k_ in v], device=dev) for s, v in SPL.items()}
print('물체 ' + ' · '.join(f'{s} {len(v)}' for s, v in SPL.items()) + f' | 범주 {len(CL)} | 피험자 {SUBS}', flush=True)
load = lambda si, k_: np.load(os.path.join(a.data_root, SUBS[si], 'npy_data', k_ + '.npy')).astype(np.float32)[FRAMES]

from omegaconf import OmegaConf
enc = load_encoder(a.enc_path, OmegaConf.load(os.path.join(a.repo, 'configs', 'mind3d_pp.yaml'))).to(dev).eval().requires_grad_(False)
cal = RegionCalib(K).to(dev); c = torch.load(a.calib, map_location='cpu'); assert list(c['subs']) == SUBS
cal.g.data.copy_(c['g']); cal.b.data.copy_(c['b']); cal.requires_grad_(False)
print(f'보정 {a.calib} | |g| {float(c["g"].abs().mean()):.3f} |b| {float(c["b"].abs().mean()):.3f} | {c.get("summary", "")}', flush=True)


def norm_in(x):
    if not a.in_norm: return x
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


@torch.no_grad()
def vit(x):                                                     # x [B, T, 256, 256] → 프레임별 [B, T, 64·1024]
    B, T = x.shape[:2]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        t = enc.forward_encoder_wo_pred(x.reshape(B * T, 1, 256, 256)).view(B, T, 257, -1)[:, :, 1:].float()
    return F.layer_norm(t, t.shape[-1:]).view(B, T, 8, 2, 8, 2, -1).mean((3, 5)).flatten(2)


@torch.no_grad()
def collect(split, si, on):
    """특징 [N, D] (프레임 6 평균), 앞 3 · 뒤 3 평균, 입력 변화 비율"""
    Z, Za, Zb, dx = [], [], [], []
    for i in range(0, len(SPL[split]), 16):
        x = norm_in(torch.from_numpy(np.stack([load(si, k_) for k_ in SPL[split][i:i + 16]])).to(dev))
        if on:
            x2 = cal(x, torch.full((len(x),), si, dtype=torch.long, device=dev)); dx.append(((x2 - x).flatten(1).norm(dim=1) / x.flatten(1).norm(dim=1)).cpu()); x = x2
        f = vit(x); Z.append(f.mean(1).half()); Za.append(f[:, :3].mean(1).half()); Zb.append(f[:, 3:].mean(1).half())
    return torch.cat(Z), torch.cat(Za), torch.cat(Zb), (torch.cat(dx).mean().item() if dx else 0.0)


def center(z, mu): return F.normalize(z.float() - mu, dim=-1)


def ridge_acc(Ztr, ytr, Zev, yev, lam):
    Kt = Ztr @ Ztr.t(); T_ = F.one_hot(ytr, len(CL)).float()
    A = torch.linalg.solve(Kt + lam * torch.eye(len(Kt), device=dev), T_)
    return float(((Zev @ Ztr.t() @ A).argmax(-1) == yev).float().mean())


R = {}; t0 = time.time()
for on in (0, 1):
    tag = '보정 후' if on else '보정 전'; F_ = {}
    for si in range(K):
        for s in ('train', 'hold', 'test'): F_[si, s] = collect(s, si, on)
        print(f'  {tag} {SUBS[si]} 특징 끝 {time.time() - t0:.0f}s', flush=True)
    mu = {si: F_[si, 'train'][0].float().mean(0) for si in range(K)}
    C = {(si, s): center(F_[si, s][0], mu[si]) for si in range(K) for s in ('train', 'hold', 'test')}
    r = {'input_change': {SUBS[si]: round(F_[si, 'test'][3], 4) for si in range(K)}}
    r['obj_var_ratio'] = {SUBS[si]: round(float((F_[si, 'test'][0].float() - mu[si]).norm(dim=1).mean() / F_[si, 'test'][0].float().norm(dim=1).mean()), 4) for si in range(K)}
    # ① 범주 선형 읽기 (λ는 hold로)
    cat = {}
    for si in range(K):
        for sj in range(K):
            best = max((ridge_acc(C[si, 'train'], Y['train'], C[sj, 'hold'], Y['hold'], l), l) for l in (0.01, 0.1, 1, 10, 100))
            cat[f'{SUBS[si]}→{SUBS[sj]}'] = {'test': round(ridge_acc(C[si, 'train'], Y['train'], C[sj, 'test'], Y['test'], best[1]), 4), 'hold': round(best[0], 4), 'lam': best[1]}
    r['category_probe'] = cat
    # ② 같은 피험자 안 물체 구별 (앞 3 프레임 ↔ 뒤 3 프레임)
    r['within_subject_obj_top1'] = {}
    for si in range(K):
        A_, B_ = center(F_[si, 'test'][1], mu[si]), center(F_[si, 'test'][2], mu[si])
        r['within_subject_obj_top1'][SUBS[si]] = round(float(((A_ @ B_.t()).argmax(-1) == torch.arange(len(A_), device=dev)).float().mean()), 4)
    # ③ 다른 피험자 같은 물체 고르기 (test)
    Zc = torch.cat([C[si, 'test'] for si in range(K)]); S = torch.arange(K, device=dev).repeat_interleave(len(SPL['test'])); O = torch.arange(len(SPL['test']), device=dev).repeat(K)
    sim = Zc @ Zc.t(); other = S[:, None] != S[None]; same = (O[:, None] == O[None]) & other
    r['cross_subject_obj_top1'] = round(float((O[sim.masked_fill(~other, -2).argmax(-1)] == O).float().mean()), 4)
    r['cross_subject_gap'] = round(float(sim[same].mean() - sim[other & ~same].mean()), 4)
    R['on' if on else 'off'] = r; R[f'feat_{"on" if on else "off"}'] = None
    if on:                                                      # ④ 보정 전후 특징 cos (test, 원 특징)
        R['feat_cos_before_after'] = {SUBS[si]: round(float(F.cosine_similarity(F_[si, 'test'][0].float(), Foff[si].float(), dim=-1).mean()), 4) for si in range(K)}
    else:
        Foff = {si: F_[si, 'test'][0] for si in range(K)}
    print(f'[{tag}]', json.dumps(r, ensure_ascii=False), flush=True)
R = {k: v for k, v in R.items() if v is not None}; R['chance'] = {'category': round(1 / len(CL), 4), 'obj_test': round(1 / len(SPL['test']), 4)}
json.dump(R, open(a.out, 'w'), ensure_ascii=False, indent=1); print('DIAG_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
