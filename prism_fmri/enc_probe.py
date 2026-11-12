# 인코더 탐침 (11-12): 고정 fMRI ViT 출력에서 범주를 선형 해독 — "덜 외운 인코더"(UKB 사전학습, fMRI-Shape를 본 적 없음)와 공개 피험자별 인코더 비교
#  --enc ukb=<경로> : 피험자 공통 가중치 하나 (encoder only, 키 fmri_encoder.*, clip_norm · aggregation 없음 → 그 층은 안 씀)
#  --enc sub=<폴더> : <폴더>/subNN_encoder.pt (MinD-3D++ 공개, 지금 쓰는 것)
#  특징: forward_encoder_wo_pred, 프레임 2~7 평균, 패치 256 × 1024 (pos) / 패치 평균 1024 (mean)
#  해독기: roi_decode.py fit() 그대로 (선형 커널 ridge, alpha LOO), 차원마다 맞춘 물체로 표준화
#  값: ① 검증 물체(train 범주마다 목록 끝 --hold개)를 빼고 맞춤 → 검증 물체 / test   ② train 1300 전부로 맞춤 → test
#  --znorm 1: 입력을 프레임마다 0 아닌 픽셀 기준 z-score (UKB 사전학습 전처리가 다를 가능성 확인용)
# 실행 위치 $W/run (MinD-3D++ venv)
import os, sys, json, argparse, numpy as np, torch
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from src.encoder.autoencoder2 import fMRI_Encoder

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--enc', required=True)
p.add_argument('--subs', default='sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008'); p.add_argument('--out', required=True)
p.add_argument('--hold', type=int, default=8); p.add_argument('--bs', type=int, default=16); p.add_argument('--znorm', type=int, default=0)
a = p.parse_args(); os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
cfg = OmegaConf.load(os.path.join(a.repo, 'configs', 'mind3d_pp.yaml')); kind, path = a.enc.split('=', 1); FRAMES = list(range(2, 8))
rd = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
TR, TE = rd('train'), rd('test'); SYN = sorted({k.split('/')[0] for k in TR}); TE = [k for k in TE if k.split('/')[0] in SYN]
by = {}
for k in TR: by.setdefault(k.split('/')[0], []).append(k)
Hs = {k for v in by.values() for k in v[-a.hold:]}; HO = [k for k in TR if k in Hs]; FIT = [k for k in TR if k not in Hs]
ALL = FIT + HO + TE; nf, nh = len(FIT), len(HO); cls = np.array([SYN.index(k.split('/')[0]) for k in ALL]); Yc = np.eye(len(SYN))[cls]
ALPHA_REL = [10.0 ** k for k in np.arange(-3, 3.01, 0.5)]


def load(pth):
    e = fMRI_Encoder(cfg.model.params.fmri_encoder_config); sd = torch.load(pth, map_location='cpu'); sd = sd.get('model', sd)
    r = e.load_state_dict({k[len('fmri_encoder.'):]: v for k, v in sd.items() if k.startswith('fmri_encoder.')}, strict=False)
    print(f'[enc] {pth} | 없는 키 {sorted({k.split(".")[0] for k in r.missing_keys})} · 남는 키 {len(r.unexpected_keys)}', flush=True)
    return e.cuda().eval()


def fit(Kt, Y):   # roi_decode.py fit() 그대로
    rm = Kt.mean(0); am = Kt.mean(); s, U = np.linalg.eigh(Kt - rm[None] - rm[:, None] + am); s = np.clip(s, 0, None)
    ym = Y.mean(0); UtY = U.T @ (Y - ym); U2 = U ** 2; err = np.zeros((len(ALPHA_REL), Y.shape[1])); n = len(Kt)
    for ai, r in enumerate(ALPHA_REL):
        h = s / (s + r * s.mean()); d = np.clip(1 - U2 @ h - 1 / n, 1e-6, None)
        err[ai] = ((((Y - ym) - U @ (h[:, None] * UtY)) / d[:, None]) ** 2).sum(0)
    al = np.array(ALPHA_REL)[err.argmin(0)] * s.mean(); C = U @ (UtY / (s[:, None] + al[None]))
    return lambda Ke: (Ke - rm[None] - Ke.mean(1, keepdims=True) + am) @ C + ym


def gram(X, ix):  # X [N, D] → K [N, len(ix)], 차원마다 ix 물체로 표준화
    N, D = X.shape; K = torch.zeros(N, len(ix), device='cuda', dtype=torch.float64); ixt = torch.from_numpy(ix).cuda()
    for j in range(0, D, 8192):
        c = X[:, j:j + 8192].float(); mu = c[ixt].mean(0); sd = c[ixt].std(0) + 1e-6; c = (c - mu) / sd
        K += (c @ c[ixt].T).double()
    return (K / D).cpu().numpy()


def acc(X, fit_ix, ev):   # ev: {이름: 인덱스}
    K = gram(X, fit_ix); f = fit(K[fit_ix], Yc[fit_ix])
    return {n: float((f(K[ix]).argmax(1) == cls[ix]).mean()) for n, ix in ev.items()}


I_F, I_H, I_T = np.arange(nf), np.arange(nf, nf + nh), np.arange(nf + nh, len(ALL)); R = {}
enc = load(path) if kind == 'ukb' else None
for sub in a.subs.split(','):
    e = enc if enc is not None else load(f'{path}/sub{sub[-2:]}_encoder.pt'); F = []
    for i in range(0, len(ALL), a.bs):
        x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ALL[i:i + a.bs]])).cuda()
        B, T = x.shape[:2]
        if a.znorm:
            m = (x != 0).float(); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
            sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; x = (x - mu) / sd * m
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            t = e.forward_encoder_wo_pred(x.reshape(B * T, 1, 256, 256)).view(B, T, 257, -1).float()
        F.append(t[:, :, 1:].mean(1).half())
    F = torch.cat(F); r = {}
    for nm, X in (('pos', F.flatten(1)), ('mean', F.float().mean(1).half())):
        h = acc(X, I_F, {'hold': I_H, 'test': I_T}); full = acc(X, np.concatenate([I_F, I_H]), {'test': I_T})
        r[nm] = {'hold': h['hold'], 'test': h['test'], 'test_fit_all': full['test']}
    R[sub] = r; print(f'P| {sub} ' + ' '.join(f'{nm} 검증 {v["hold"]:.3f} / test {v["test"]:.3f} (train 전부로 맞춤 {v["test_fit_all"]:.3f})' for nm, v in r.items()), flush=True)
    if enc is None: del e
    del F; torch.cuda.empty_cache(); json.dump({'enc': a.enc, 'znorm': a.znorm, 'per_sub': R}, open(a.out, 'w'), indent=1)
print('P| 평균 ' + ' '.join(f'{nm} 검증 {np.mean([R[s][nm]["hold"] for s in R]):.3f} / test {np.mean([R[s][nm]["test"] for s in R]):.3f} (train 전부 {np.mean([R[s][nm]["test_fit_all"] for s in R]):.3f})' for nm in ('pos', 'mean')), flush=True)
print('PROBE_DONE', flush=True)
