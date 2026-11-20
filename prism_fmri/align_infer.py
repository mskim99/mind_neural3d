# -*- coding: utf-8 -*-
# 11-20 Perceptive 1단계 (새 피험자 붙이기): 학습에 없던 피험자 B의 fMRI를 학습된 피험자 A의 입력 공간으로 옮기는 선형 사상(기능 정렬)을 같은 물체 N개 짝으로 맞춘 뒤,
#  A의 갈래(고정, 추가 학습 없음)로 그대로 추론 — N을 바꿔 가며 범주 정확도를 잼 (N-shot 곡선). 진단 diag_align.py(선형 분류기)의 실제 모델판
#  사상: 프레임 단위 커널 릿지 (입력 = in_norm 뒤 평면 지도의 0 아닌 픽셀, 프레임별 평균을 빼고 맞춤). 짝에는 범주 라벨이 필요 없음 (같은 자극을 본 fMRI만)
#   λ: 짝의 80 %로 맞추고 나머지 20 %에서 A의 실제 패턴과의 상관이 가장 높은 값 (라벨 · test 안 씀). 그 뒤 N개 전부로 다시 맞춤
#   gain: 릿지는 물체별 변화를 줄여서 내므로, 같은 20 %에서 A의 실제 변화 크기에 맞춘 배율을 곱한 판도 같이 냄 (g1 = 배율 없음, gm = 배율 맞춤)
#  N = 0: 정렬 없이 B의 fMRI를 그대로 A 갈래에 (기준). 짝 물체 = core_train 목록에서 고정 순서(seed 0)로 앞 N개. test 104개는 사상에 안 씀
#  출력: <out>/align.json (쌍 · N · gain별 범주 정확도, 고른 λ, 20 % 상관) + --save "N:gain,…"이면 <out>/n<N>_<gain>/<B>_as<A 번호>_keep/{tokens, components.npz} (infer_prism.py와 같은 형식)
# 사용 (실행 위치 $W/run): python align_infer.py --repo $W/MinD-3D --data_root $DR --ckpt $W/prism_2s/ausvp/model_final.pt --subs sub-0001,sub-0006 --fmri_subs sub-0002,sub-0003 --out $W/prism_align/ausvp
import os, sys, json, time, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
p.add_argument('--subs', required=True); p.add_argument('--fmri_subs', required=True); p.add_argument('--ns', default='0,50,100,300,600,1300')
p.add_argument('--lams', default='0.01,0.03,0.1,0.3,1,3,10,30'); p.add_argument('--save', default=''); p.add_argument('--bs', type=int, default=8); p.add_argument('--self_ref', type=int, default=1)
p.add_argument('--fit', default='frame', choices=['frame', 'mean'])   # frame: (물체, 프레임)을 표본으로 / mean: 물체의 프레임 평균 패턴을 표본으로 맞춘 사상을 프레임마다 적용 (잡음이 적음, diag_align.py와 같은 단위)
a = p.parse_args()
dev = 'cuda'; t0 = time.time(); ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8)); T = len(FRAMES)
model = build_from_ckpt(ck, a.repo).cuda().eval()
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
TR, TE = lst('train'), lst('test'); perm = torch.randperm(len(TR), generator=torch.Generator().manual_seed(0)).tolist(); TRP = [TR[i] for i in perm]
yte = np.array([SYN.index(k.split('/')[0]) for k in TE]); SAVE = {tuple(s.split(':')) for s in a.save.split(',') if s}
os.makedirs(a.out, exist_ok=True)


def norm_in(x):                                                             # 모델의 in_norm과 같은 식 (프레임마다 0 아닌 픽셀 z-score)
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


def raw(s, keys):                                                           # [n, T, 256, 256] (원 값, cpu)
    return torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in keys]))


_x = raw(a.subs.split(',')[0], TE[:4]); M = (_x != 0).any(0).any(0).flatten().to(dev); P = int(M.sum())
def feat(s, keys):                                                          # [n, T, P] in_norm 뒤 0 아닌 픽셀
    o = []
    for i in range(0, len(keys), 64): o.append(norm_in(raw(s, keys[i:i + 64]).to(dev)).flatten(2)[:, :, M])
    return torch.cat(o)
def unflat(z):                                                              # [n, T, P] → [n, T, 256, 256]
    o = torch.zeros(*z.shape[:2], 256 * 256, device=dev); o[:, :, M] = z; return o.view(*z.shape[:2], 256, 256)


@torch.no_grad()
def run(x, si, od=None):                                                    # x [n, T, 256, 256] → 범주 예측 (+ 토큰 · components 저장)
    pr, comp = [], {k: [] for k in ('sem', 'var', 'perc', 'cls_score', 'clip', 'vt')}
    if od: os.makedirs(f'{od}/tokens', exist_ok=True)
    for i in range(0, len(x), a.bs):
        xb = x[i:i + a.bs].to(dev); fi = torch.tensor(FRAMES, device=dev).expand(len(xb), -1)
        with torch.autocast('cuda', dtype=torch.bfloat16): out = model(xb, fi, torch.full((len(xb),), si, dtype=torch.long, device=dev))
        sc = (out['sem_clip'].float() @ model.prototypes.t()).cpu(); pr.append(sc.argmax(-1))
        if od:
            for j, k in enumerate(TE[i:i + a.bs]): np.save(f'{od}/tokens/{k.split("/")[1]}.npy', out['cond_tokens'][j].float().half().cpu().numpy())
            comp['sem'].append(out['sem_vec'].float().cpu()); comp['var'].append(out['var_vec'].float().cpu()); comp['perc'].append(out['perc_vec'].float().cpu()); comp['cls_score'].append(sc); comp['clip'].append(out['sem_clip'].float().cpu())
            if 'vt' in out: comp['vt'].append(out['vt'].float().mean(1).cpu())
    pr = torch.cat(pr).numpy()
    if od: np.savez(f'{od}/components.npz', name=np.array(TE), cls=yte, classes=np.array(SYN), cls_pred=pr, **{k: torch.cat(v).numpy() for k, v in comp.items() if v})
    return float((pr == yte).mean())


def solve(K, Y, lam): return torch.linalg.solve(K.double() + lam * K.diagonal().mean().double() * torch.eye(len(K), device=dev, dtype=torch.float64), Y.double()).float()
def rcorr(A, B): A = A - A.mean(1, keepdim=True); B = B - B.mean(1, keepdim=True); return float((torch.nn.functional.normalize(A, dim=1) * torch.nn.functional.normalize(B, dim=1)).sum(1).mean())
NSL = [int(v) for v in a.ns.split(',')]; LAMS = [float(v) for v in a.lams.split(',')]; R = {'ckpt': a.ckpt, 'px': P, 'chance': round(1 / len(SYN), 4), 'pairs': {}}
for A in a.subs.split(','):
    si = SUBS.index(A); FA = feat(A, TRP[:max(NSL)]); XAte = raw(A, TE)
    own = run(XAte, si) if a.self_ref else None; print(f'[{A}] 자기 test 범주 정확도 {own}', flush=True)
    for B in a.fmri_subs.split(','):
        if B == A: continue
        FB = feat(B, TRP[:max(NSL)]); FBte = feat(B, TE); r = {'own_A': own, 'n': {}}
        for N in NSL:
            if N == 0:
                acc = run(raw(B, TE), si, f'{a.out}/n0_g1/{B}_as{A[-2:]}_keep' if ('0', 'g1') in SAVE else None); r['n']['0'] = {'g1': round(acc, 4)}
                print(f'[{A} ← {B}] N 0 (정렬 없음) 범주 {acc:.3f}', flush=True); continue
            xa, xb = FA[:N], FB[:N]; mua, mub = xa.mean(0, keepdim=True), xb.mean(0, keepdim=True)        # 프레임별 평균 [1, T, P]
            da, db = ((xa - mua).flatten(0, 1), (xb - mub).flatten(0, 1)) if a.fit == 'frame' else ((xa - mua).mean(1), (xb - mub).mean(1))   # [N·T, P] 또는 [N, P]
            nf = max(int(N * 0.8), 1) * (T if a.fit == 'frame' else 1); best = None                                                    # λ · gain: 앞 80 % 물체로 맞추고 뒤 20 %에서
            Kf = db[:nf] @ db[:nf].t(); Kc = db[nf:] @ db[:nf].t()
            for lam in LAMS:
                pc = Kc @ solve(Kf, da[:nf], lam); c = rcorr(pc, da[nf:])
                if best is None or c > best[0]: best = (c, lam, float(da[nf:].pow(2).mean().sqrt() / pc.pow(2).mean().sqrt().clamp_min(1e-8)))
            c, lam, g = best; Wd = solve(db @ db.t(), da, lam)                                             # 전부로 다시 [N·T, P]
            dz = ((FBte - mub).flatten(0, 1) @ db.t() @ Wd).view(len(TE), T, P); e = {'lam': lam, 'corr20': round(c, 4), 'gain': round(g, 3)}
            for tag, gg in (('g1', 1.0), ('gm', g)):
                od = f'{a.out}/n{N}_{tag}/{B}_as{A[-2:]}_keep' if (str(N), tag) in SAVE else None
                e[tag] = round(run(unflat(mua + gg * dz).cpu(), si, od), 4)
            r['n'][str(N)] = e; print(f'[{A} ← {B}] N {N} λ {lam:g} 20 % 상관 {c:.3f} gain {g:.2f} | 범주 g1 {e["g1"]:.3f} · gm {e["gm"]:.3f} | {time.time() - t0:.0f}s', flush=True)
            del Wd, dz, Kf, Kc; torch.cuda.empty_cache()
        R['pairs'][f'{A}<-{B}'] = r; json.dump(R, open(f'{a.out}/align.json', 'w'), ensure_ascii=False, indent=1); del FB, FBte
    del FA; torch.cuda.empty_cache()
# 요약: N별 쌍 평균
S = {}
for N in NSL:
    for tag in ('g1', 'gm'):
        v = [r['n'][str(N)][tag] for r in R['pairs'].values() if tag in r['n'].get(str(N), {})]
        if v: S[f'n{N}_{tag}'] = round(float(np.mean(v)), 4)
R['mean'] = S; R['own_mean'] = round(float(np.mean([r['own_A'] for r in R['pairs'].values() if r['own_A'] is not None])), 4) if a.self_ref else None
json.dump(R, open(f'{a.out}/align.json', 'w'), ensure_ascii=False, indent=1); print('쌍 평균', json.dumps(S), '| 자기', R['own_mean'], flush=True); print('ALIGN_INFER_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
