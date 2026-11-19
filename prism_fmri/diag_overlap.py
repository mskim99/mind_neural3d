# -*- coding: utf-8 -*-
# 11-19 분석: 피험자 7명의 활성 지도가 평면 지도(256×256) 위에서 얼마나 겹치나 — 그림용 (학습 없음)
#  활성 지도: 피험자마다 fMRI 프레임 2~7 평균 (in_norm). 세 수준으로 봄
#    전체   모든 물체 평균 반응 (그 피험자의 평균 활성)
#    범주별 범주 평균 반응 − 그 피험자의 전체 평균 (그 범주에서만 더 켜지는 곳)
#    물체별 물체 반응 − 그 피험자의 전체 평균 (그 물체에서만 더 켜지는 곳)
#  겹침: 지도마다 값이 큰 위 20 % 픽셀을 "켜짐"으로 → 픽셀마다 켜진 피험자 수(0~7) / 피험자 쌍마다 켜진 영역의 Dice (우연 = 0.20)
#  + 픽셀 단위 피험자 간 상관 (지도 전체 픽셀 값의 상관, 쌍 평균)
#  출력 npz: mask, cnt_all [256,256], cnt_cat [13,256,256], cnt_obj [K,256,256](test 물체 K개 예시) · obj_names, dice · corr 통계, 영역별 Dice
# 사용 (실행 위치 $W/run): python diag_overlap.py --data_root $DR --subs ... --out $W/prism_calib/overlap.npz
import os, sys, json, time, argparse, itertools, numpy as np, torch
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True); p.add_argument('--subs', required=True); p.add_argument('--out', required=True)
p.add_argument('--top', type=float, default=0.2); p.add_argument('--n_ex', type=int, default=8)
a = p.parse_args()
dev = 'cuda'; t0 = time.time(); SUBS = a.subs.split(','); FRAMES = list(range(2, 8)); NS = len(SUBS)
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
has = lambda k_: all(os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')) for s in SUBS)
KEYS = [k_ for k_ in lst('train') + lst('test') if has(k_)]; TEST = set(lst('test'))
CL = sorted({k_.split('/')[0] for k_ in KEYS}); Y = torch.tensor([CL.index(k_.split('/')[0]) for k_ in KEYS], device=dev)
here = os.path.dirname(os.path.abspath(__file__))
lab = torch.as_tensor(np.load(os.path.join(here, 'assets', 'mmp_256_regions.npy'))).long().to(dev).flatten(); M = lab > 0; P = int(M.sum())
NAME = {int(k): v for k, v in json.load(open(os.path.join(here, 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
print(f'물체 {len(KEYS)} · 범주 {len(CL)} · 피험자 {NS} · 픽셀 {P}', flush=True)


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


@torch.no_grad()
def collect(s):
    o = []
    for i in range(0, len(KEYS), 32):
        x = norm_in(torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')).astype(np.float32)[FRAMES] for k_ in KEYS[i:i + 32]])).to(dev))
        o.append(x.mean(1).flatten(1)[:, M].half())
    return torch.cat(o)                                                    # [N, P]


X = [collect(s) for s in SUBS]; print(f'읽기 끝 {time.time() - t0:.0f}s', flush=True)
k = int(P * a.top)
def on(m):                                                                 # [..., P] → 위 top 비율 픽셀 True
    th = m.float().kthvalue(P - k + 1, dim=-1, keepdim=True).values; return m.float() >= th
PAIRS = list(itertools.combinations(range(NS), 2))
def dice(B):                                                               # B [S, n, P] bool → 쌍 평균 Dice [n]
    return torch.stack([(B[i] & B[j]).float().sum(-1) / k for i, j in PAIRS]).mean(0)
def corr(A):                                                               # A [S, n, P] → 쌍 평균 픽셀 상관 [n]
    Z = A.float() - A.float().mean(-1, keepdim=True); Z = Z / Z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.stack([(Z[i] * Z[j]).sum(-1) for i, j in PAIRS]).mean(0)
mean_all = torch.stack([x.float().mean(0) for x in X])                     # [S, P]
cat_m = torch.stack([torch.stack([x[Y == c].float().mean(0) for c in range(len(CL))]) for x in X]) - mean_all[:, None]   # [S, 13, P]
B_all = on(mean_all)[:, None]; B_cat = on(cat_m)
R = {'top': a.top, 'chance_dice': a.top, 'dice_all': float(dice(B_all)[0]), 'corr_all': float(corr(mean_all[:, None])[0]),
     'dice_cat': dice(B_cat).cpu().numpy(), 'corr_cat': corr(cat_m).cpu().numpy()}
ti = torch.tensor([i for i, k_ in enumerate(KEYS) if k_ in TEST], device=dev)
d_obj, c_obj = [], []
for i0 in range(0, len(ti), 26):
    idx = ti[i0:i0 + 26]; O = torch.stack([x[idx].float() for x in X]) - mean_all[:, None]       # [S, n, P]
    d_obj.append(dice(on(O))); c_obj.append(corr(O))
R['dice_obj'] = torch.cat(d_obj).cpu().numpy(); R['corr_obj'] = torch.cat(c_obj).cpu().numpy()
# 물체 − 범주 평균 (같은 범주 안 차이)
d_w, c_w = [], []
for i0 in range(0, len(ti), 26):
    idx = ti[i0:i0 + 26]; O = torch.stack([x[idx].float() for x in X]) - mean_all[:, None] - cat_m[:, Y[idx]]
    d_w.append(dice(on(O))); c_w.append(corr(O))
R['dice_objwc'] = torch.cat(d_w).cpu().numpy(); R['corr_objwc'] = torch.cat(c_w).cpu().numpy()
def full(v):                                                               # [.., P] → [.., 256, 256]
    o = torch.zeros(*v.shape[:-1], 256 * 256, device=dev); o[..., M] = v.float(); return o.view(*v.shape[:-1], 256, 256).cpu().numpy()
ex = []; seen = set()
for i in ti.tolist():                                                      # 범주마다 첫 test 물체 → 앞 n_ex개
    c = int(Y[i])
    if c not in seen: seen.add(c); ex.append(i)
ex = ex[:a.n_ex]; O = torch.stack([x[ex].float() for x in X]) - mean_all[:, None]
# 영역별 Dice (영역 안 픽셀에서 켜진 것끼리, 전체 · 범주 평균)
REG = {}
for r, n in NAME.items(): REG.setdefault(n.split('_', 1)[-1], []).append(r)
labm = lab[M]; reg = {}
for n, ids in REG.items():
    m = torch.isin(labm, torch.tensor(ids, device=dev))
    if int(m.sum()) < 40: continue
    def rd(B):
        return float(torch.stack([(2 * (B[i][..., m] & B[j][..., m]).float().sum(-1) / (B[i][..., m].float().sum(-1) + B[j][..., m].float().sum(-1)).clamp_min(1)) for i, j in PAIRS]).mean())
    reg[n] = {'px': int(m.sum()), 'on_all': float(B_all[:, 0][:, m].float().mean()), 'dice_all': rd(B_all), 'dice_cat': rd(B_cat),
              'corr_cat': float(corr(cat_m[..., m]).mean())}
np.savez_compressed(a.out, mask=M.view(256, 256).cpu().numpy(), lab=lab.view(256, 256).cpu().numpy(), cnt_all=full(B_all[:, 0].float().sum(0)), cnt_cat=full(B_cat.float().sum(0)),
                    cnt_obj=full(on(O).float().sum(0)), obj_names=np.array([KEYS[i] for i in ex]), classes=np.array(CL), subs=np.array(SUBS),
                    mean_sub=full(mean_all), reg=json.dumps(reg, ensure_ascii=False), **{k_: np.asarray(v) for k_, v in R.items()})
print(json.dumps({k_: (round(float(np.mean(v)), 4) if not isinstance(v, float) else round(v, 4)) for k_, v in R.items()}, ensure_ascii=False))
print('OVERLAP_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
