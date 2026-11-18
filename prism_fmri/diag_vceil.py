# -*- coding: utf-8 -*-
# 11-18 진단: fMRI에 "같은 범주 안 형상 차이"(Variational이 맡을 정보)가 선형으로 얼마나 읽히나 — 학습 없음, 커널 릿지만
#  V 개선안(V 잔차 · 공통 S·V)이 모두 실패한 뒤, 구조를 더 바꿀 가치가 있는지 상한을 잼
#  입력 (피험자마다, 프레임 2~7 평균, in_norm 같은 식):
#    raw:<영역묶음>  원 fMRI 평면 지도에서 그 영역 픽셀만 (HCP-MMP, assets/mmp_256_regions.*)
#    vit             고정 UKB ViT 특징 (diag_calib.py와 같음: LayerNorm → 2×2 묶음 64개 → 폄)
#  정답:
#    res   GT 토큰(자극 프레임 0 DINOv2)의 범주 평균 뺀 잔차 PCA 32 계수 (vres_basis_k32.npz, 표준화)
#    sil   현재 V head 정답 = 물체 평균 실루엣 [cy, w, h, area] − 범주 평균 (view_targets_shape.npz)
#    map   실루엣 지도 32×32 (영상 10프레임 평균) / vox (11-19 V-3) 3D 점유 격자 64³ → 4×4×4 평균 16³ = 4,096 (voxels/, 없는 물체는 빼고 잼)
#  --maps_out <폴더> (11-19 V-0 대조): 실루엣 지도 2벌 저장 — sil_catmean.npz(map: 정답 범주의 train 평균 지도) · sil_pred.npz(map_<피험자>: 범주 평균 + V1–V4 원 신호 커널 릿지 예측 잔차, train · hold는 5겹 밖 예측)
#  범주 안 정보만 보려고 특징 · 정답 모두 정답 범주의 train 평균을 뺌 (범주는 맞혔다고 치는 상한)
#  지표 (test): res → 같은 범주 test 물체 중 정답 고르기 top-1 (우연 ≈ 1/8), 계수 R²(평균) / sil → 성분별 R² · 상관
#  λ는 hold로 고름 (res: 고르기 top-1, sil: R²)
# 사용 (실행 위치 $W/run): python diag_vceil.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs sub-0001,sub-0006 --enc_path $U --out $W/prism_calib/vceil.json
import os, sys, json, time, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import load_encoder

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--subs', required=True); p.add_argument('--enc_path', required=True); p.add_argument('--holdout', type=int, default=8)
p.add_argument('--out', required=True); p.add_argument('--limit', type=int, default=0); p.add_argument('--maps_out', default='')
a = p.parse_args()
dev = 'cuda'; FRAMES = list(range(2, 8)); SUBS = a.subs.split(','); t0 = time.time()
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
tr_all = lst('train'); by = {}
for k_ in tr_all: by.setdefault(k_.split('/')[0], []).append(k_)
H = {k_ for v in by.values() for k_ in v[-a.holdout:]}
VR = np.load(f'{a.tgt}/vres_basis_k32.npz'); RES = {str(k).split('/')[-1]: VR['shape'][i, 0] for i, k in enumerate(VR['keys'])}
VS = np.load(f'{a.tgt}/view_targets_shape.npz'); SIL = {str(k).split('/')[-1]: (VS['shape'][i, 0], bool(VS['shape_ok'][i])) for i, k in enumerate(VS['keys'])}
SM = np.load(f'{a.tgt}/silhouettes32.npz'); MAP = {str(u): SM['sil'][i].astype(np.float32).mean(0) / max(1.0, float(SM['sil'].max())) for i, u in enumerate(SM['uids'])}   # 영상 10프레임 평균 실루엣 32×32
uid = lambda k_: k_.split('/')[1]
def _vox(u):
    f = f'{a.tgt}/voxels/{u}.npy'
    if not os.path.exists(f): return np.zeros(16 ** 3, np.float32), False
    o = np.unpackbits(np.load(f))[:64 ** 3].reshape(16, 4, 16, 4, 16, 4).astype(np.float32).mean((1, 3, 5)); return o.ravel(), True
has = lambda k_: uid(k_) in RES and uid(k_) in MAP and uid(k_) in SIL and all(os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')) for s in SUBS)
SPL = {'train': [k_ for k_ in tr_all if k_ not in H and has(k_)], 'hold': [k_ for k_ in tr_all if k_ in H and has(k_)], 'test': [k_ for k_ in lst('test') if has(k_)]}
if a.limit: SPL = {s: v[:a.limit] for s, v in SPL.items()}
CL = sorted(by); Y = {s: torch.tensor([CL.index(k_.split('/')[0]) for k_ in v], device=dev) for s, v in SPL.items()}
TR = {s: torch.tensor(np.stack([RES[uid(k_)] for k_ in v]), device=dev).float() for s, v in SPL.items()}
TS = {s: torch.tensor(np.stack([SIL[uid(k_)][0] for k_ in v]), device=dev).float() for s, v in SPL.items()}
TM = {s: torch.tensor(np.stack([MAP[uid(k_)].ravel() for k_ in v]), device=dev).float() for s, v in SPL.items()}
OK = {s: torch.tensor([SIL[uid(k_)][1] for k_ in v], device=dev) for s, v in SPL.items()}
_VX = {s: [_vox(uid(k_)) for k_ in v] for s, v in SPL.items()}
TV = {s: torch.tensor(np.stack([x for x, _ in v]), device=dev) for s, v in _VX.items()}; OKV = {s: torch.tensor([o for _, o in v], device=dev) for s, v in _VX.items()}
print('복셀 있는 물체 ' + ' · '.join(f'{s} {int(v.sum())}' for s, v in OKV.items()), flush=True)
print('물체 ' + ' · '.join(f'{s} {len(v)}' for s, v in SPL.items()) + f' | 범주 {len(CL)}', flush=True)

lab = torch.as_tensor(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mmp_256_regions.npy'))).long().to(dev)
NAME = {int(k): v for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
GROUP = {'초기시각 V1-V4': {'V1', 'V2', 'V3', 'V4'},
         '등쪽 V3A·V3B·V6·V7·IPS1·MT·MST': {'V3A', 'V3B', 'V6', 'V6A', 'V7', 'IPS1', 'MT', 'MST', 'LO3'},
         '배쪽·LO LO1·LO2·PIT·FFC·V8·VVC·PH': {'LO1', 'LO2', 'PIT', 'FFC', 'V8', 'VVC', 'PH', 'VMV1', 'VMV2', 'VMV3', 'TE2p', 'TF'}}
gmask = {g: torch.isin(lab, torch.tensor([r for r, n in NAME.items() if n.split('_', 1)[-1] in names], device=dev)) for g, names in GROUP.items()}
gmask['전체'] = lab > 0
for g, m in gmask.items(): print(f'  영역 {g}: 픽셀 {int(m.sum())}', flush=True)

from omegaconf import OmegaConf
enc = load_encoder(a.enc_path, OmegaConf.load(os.path.join(a.repo, 'configs', 'mind3d_pp.yaml'))).to(dev).eval().requires_grad_(False)


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


@torch.no_grad()
def collect(split, s):
    raw, vt = [], []
    for i in range(0, len(SPL[split]), 16):
        x = norm_in(torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')).astype(np.float32)[FRAMES] for k_ in SPL[split][i:i + 16]])).to(dev))
        raw.append(x.mean(1).half())
        B, T = x.shape[:2]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            t = enc.forward_encoder_wo_pred(x.reshape(B * T, 1, 256, 256)).view(B, T, 257, -1)[:, :, 1:].float()
        vt.append(F.layer_norm(t, t.shape[-1:]).view(B, T, 8, 2, 8, 2, -1).mean((3, 5)).flatten(2).mean(1).half())
    return torch.cat(raw), torch.cat(vt)


def cat_center(Xs, key):                     # 정답 범주의 train 평균 빼기 (특징 · 정답 같이)
    mu = torch.stack([Xs['train'][Y['train'] == c].float().mean(0) if (Y['train'] == c).any() else torch.zeros_like(Xs['train'][0]).float() for c in range(len(CL))])
    return {s: Xs[s].float() - mu[Y[s]] for s in Xs}


def krr(Xtr, Ttr, Xev, lam):
    Kt = Xtr @ Xtr.t(); A = torch.linalg.solve(Kt + lam * Kt.diagonal().mean() * torch.eye(len(Kt), device=dev), Ttr)
    return Xev @ Xtr.t() @ A


def within_top1(P, T_, y):
    P_, T2 = F.normalize(P, dim=-1), F.normalize(T_, dim=-1); S = P_ @ T2.t(); hit = []
    for i in range(len(P)):
        m = (y == y[i]).nonzero().squeeze(1); hit.append(float(m[S[i, m].argmax()] == i))
    return float(np.mean(hit))


def r2(P, T_, ok=None):
    if ok is not None: P, T_ = P[ok], T_[ok]
    return (1 - ((P - T_) ** 2).sum(0) / ((T_ - T_.mean(0)) ** 2).sum(0)).cpu().numpy()


LAMS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10)
R = {'chance_within': round(float(np.mean([1 / int((Y['test'] == c).sum()) for c in Y['test']])), 4)}
TRc = cat_center(TR, 'res'); TSc = cat_center(TS, 'sil'); TMc = cat_center(TM, 'map')
_TVok = {s: TV[s][OKV[s]] for s in TV}; _Yv = {s: Y[s][OKV[s]] for s in Y}
_mu = torch.stack([_TVok['train'][_Yv['train'] == c].mean(0) for c in range(len(CL))]); TVc = {s: TV[s] - _mu[Y[s]] for s in TV}   # 복셀: 있는 물체로 범주 평균
MAPS = {}
MU_M = TM['test'] - TMc['test']                                       # 정답 범주 평균 실루엣 (test)
def pcorr(A, B): A = A - A.mean(1, keepdim=True); B = B - B.mean(1, keepdim=True); return float((F.normalize(A, dim=-1) * F.normalize(B, dim=-1)).sum(-1).mean())
R['map 범주 평균만: 화소 상관'] = round(pcorr(MU_M, TM['test']), 4)
for s in SUBS:
    raw, vit = {}, {}
    for sp in SPL: raw[sp], vit[sp] = collect(sp, s)
    print(f'  {s} 특징 끝 {time.time() - t0:.0f}s', flush=True)
    feats = {f'raw:{g}': {sp: raw[sp].flatten(1)[:, m.flatten()] for sp in SPL} for g, m in gmask.items()}
    feats['vit'] = vit
    for fn, Xs in feats.items():
        Xc = cat_center(Xs, fn); Xc = {sp: F.normalize(v, dim=-1) for sp, v in Xc.items()}
        lr = max((within_top1(krr(Xc['train'], TRc['train'], Xc['hold'], l), TRc['hold'], Y['hold']), l) for l in LAMS)[1]
        P = krr(Xc['train'], TRc['train'], Xc['test'], lr)
        ok_tr, ok_h = OK['train'], OK['hold']
        ls = max((float(r2(krr(Xc['train'][ok_tr], TSc['train'][ok_tr], Xc['hold'], l), TSc['hold'], ok_h).mean()), l) for l in LAMS)[1]
        Ps = krr(Xc['train'][ok_tr], TSc['train'][ok_tr], Xc['test'], ls)
        o = OK['test']; corr = [float(np.corrcoef(Ps[o][:, j].cpu(), TSc['test'][o][:, j].cpu())[0, 1]) for j in range(Ps.shape[1])]
        r = {'res_within_top1': round(within_top1(P, TRc['test'], Y['test']), 4), 'res_r2_mean': round(float(r2(P, TRc['test']).mean()), 4),
             'res_r2_top4': [round(float(v), 3) for v in r2(P, TRc['test'])[:4]], 'lam_res': lr,
             'sil_r2[cy,w,h,area]': [round(float(v), 3) for v in r2(Ps, TSc['test'], o)], 'sil_corr': [round(c, 3) for c in corr], 'lam_sil': ls}
        lm = max((within_top1(krr(Xc['train'], TMc['train'], Xc['hold'], l), TMc['hold'], Y['hold']), l) for l in LAMS)[1]
        Pm = krr(Xc['train'], TMc['train'], Xc['test'], lm)
        r.update({'map_within_top1': round(within_top1(Pm, TMc['test'], Y['test']), 4), 'map_resid_corr': round(pcorr(Pm, TMc['test']), 4),
                  'map_full_corr(범주평균+예측)': round(pcorr(MU_M + Pm, TM['test']), 4), 'lam_map': lm})
        ov = {sp: OKV[sp] for sp in SPL}
        lv = max((within_top1(krr(Xc['train'][ov['train']], TVc['train'][ov['train']], Xc['hold'][ov['hold']], l), TVc['hold'][ov['hold']], Y['hold'][ov['hold']]), l) for l in LAMS)[1]
        Pv = krr(Xc['train'][ov['train']], TVc['train'][ov['train']], Xc['test'][ov['test']], lv)
        r.update({'vox_within_top1': round(within_top1(Pv, TVc['test'][ov['test']], Y['test'][ov['test']]), 4), 'vox_resid_corr': round(pcorr(Pv, TVc['test'][ov['test']]), 4),
                  'vox_r2_mean': round(float(r2(Pv, TVc['test'][ov['test']]).mean()), 4), 'lam_vox': lv})
        if a.maps_out and fn == 'raw:초기시각 V1-V4':                          # V-0 대조: 예측 실루엣 지도 (train · hold = 5겹 밖, test = train 전체로 맞춤)
            Xa = torch.cat([Xc['train'], Xc['hold']]); Ta = torch.cat([TMc['train'], TMc['hold']]); Pa = torch.zeros_like(Ta)
            fo = torch.arange(len(Xa), device=dev) % 5
            for f_ in range(5): Pa[fo == f_] = krr(Xa[fo != f_], Ta[fo != f_], Xa[fo == f_], lm)
            Pt = krr(Xa, Ta, Xc['test'], lm); ks = SPL['train'] + SPL['hold'] + SPL['test']
            MU_all = torch.cat([TM[sp] - TMc[sp] for sp in ('train', 'hold', 'test')])
            MAPS[s] = (ks, (MU_all + torch.cat([Pa, Pt])).clamp(0, 1).view(-1, 32, 32).cpu().numpy().astype(np.float32), MU_all.view(-1, 32, 32).cpu().numpy().astype(np.float32))
            r['maps_pred_full_corr_test'] = round(pcorr(MU_M + Pt, TM['test']), 4); r['maps_pred_full_corr_oof'] = round(pcorr(torch.cat([TM['train'] - TMc['train'], TM['hold'] - TMc['hold']]) + Pa, torch.cat([TM['train'], TM['hold']])), 4)
        R[f'{s} {fn}'] = r; print(f'[{s} {fn}]', json.dumps(r, ensure_ascii=False), flush=True)
if a.maps_out:
    os.makedirs(a.maps_out, exist_ok=True); ks0 = MAPS[SUBS[0]][0]
    assert all(MAPS[s][0] == ks0 for s in SUBS)
    np.savez_compressed(f'{a.maps_out}/sil_catmean.npz', uids=np.array([uid(k_) for k_ in ks0]), map=MAPS[SUBS[0]][2])
    np.savez_compressed(f'{a.maps_out}/sil_pred.npz', uids=np.array([uid(k_) for k_ in ks0]), **{f'map_{s}': MAPS[s][1] for s in SUBS})
    print('MAPS_SAVED', a.maps_out, len(ks0), flush=True)
json.dump(R, open(a.out, 'w'), ensure_ascii=False, indent=1); print('VCEIL_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
