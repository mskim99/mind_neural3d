# -*- coding: utf-8 -*-
# 11-28 V-10: Appearance Head (Variational 갈래 안 head) — 기존 Ours(7명 모델) 전부 고정, head만 피험자마다 학습
#  입력 : 고정 인코더 출력 토큰 [257, 1024]의 fMRI 프레임 평균 (v9_read.py의 enc와 같은 것)
#  층   : 선형 경로(토큰 펴기 → Linear) + MLP 경로(토큰 평균 → MLP×3 → LayerNorm → Linear)의 합, 출력 2 (구조도 · PoseHead와 같은 MLP 꼴)
#  정답 : 질감 통계 12개(v9_feat tex_ob)를 train(범주 평균 뺀 값 / train 표준편차)의 PCA로 돌린 PC1 · PC2,
#         정답 범주 train 평균을 빼고 train 표준편차로 나눈 값 (v9_read.py tex_pc와 같은 정의). PC1 부호 = train에서 band_hi와 양의 상관 쪽
#  학습 : train 1,300에서 범주마다 15 %를 검증으로 뗌 → 설정(weight decay · 경로)마다 조기 종료(검증 PC1 상관) → 검증 PC1 상관이 가장 큰 설정을 고름
#         비교로 선형만(커널 릿지, λ 검증으로 고름)도 같은 검증으로 잼. 고른 설정을 train 1,300 전체로 다시 학습(head는 고른 스텝 × 1300/1100)
#         test 104개는 고르는 데 쓰지 않음 (모든 설정의 test 값은 참고로만 적음)
#  출력 : <out>/head_<sub>.json (설정별 검증 · test 상관, 고른 것), head_<sub>.npz (uid, pred, true), steer_<sub>.json (α = k × clip(ŝ_PC1, ±2), V-8 쌍1 문장)
# 사용 (실행 위치 $W/run): python v10_head.py --repo $W/MinD-3D --data_root $DR --ckpt …/model_final.pt --v9 v9_feat.npz --v5 v5_feat.npz --sub sub-0001 --base_prompt …/prompt.json --out …
import os, sys, json, time, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
for k in ('repo', 'data_root', 'ckpt', 'v9', 'v5', 'sub', 'base_prompt', 'out'): p.add_argument('--' + k, required=True)
p.add_argument('--k', type=float, default=0.5); p.add_argument('--nperm', type=int, default=10000); p.add_argument('--val_frac', type=float, default=0.15)
p.add_argument('--plus', default='{an} with fine surface details and patterns'); p.add_argument('--minus', default='{an} with a smooth plain surface')
a = p.parse_args(); dev = 'cuda'; t0 = time.time(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0); torch.manual_seed(0)
ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval()
for q in model.parameters(): q.requires_grad_(False)
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
cat = lambda k: k.split('/')[0]; uid = lambda k: k.split('/')[-1]
Z9 = np.load(a.v9); Z5 = np.load(a.v5); K9 = {str(k): i for i, k in enumerate(Z9['keys'])}; K5 = {str(k): i for i, k in enumerate(Z5['keys'])}
ok = set(K9) & set(K5); TR = [k for k in lst('train') if k in ok]; TE = [k for k in lst('test') if k in ok]; CL = sorted({cat(k) for k in TR})
# ---- 정답 (v9_read.py tex_pc와 같은 식)
T_all = Z9['tex_ob'].astype(np.float64); ttr = T_all[[K9[k] for k in TR]]
cmt = {c: ttr[[i for i, k in enumerate(TR) if cat(k) == c]].mean(0) for c in CL}
rt = ttr - np.stack([cmt[cat(k)] for k in TR]); sdt = rt.std(0); _, St, Vt = np.linalg.svd(rt / sdt, full_matrices=False)
BH = Z5['feat'][:, list(Z5['names']).index('band_hi')].astype(np.float64)
pc_raw = lambda ks: (T_all[[K9[k] for k in ks]] / sdt) @ Vt[:2].T
ptr = pc_raw(TR); cmp_ = {c: ptr[[i for i, k in enumerate(TR) if cat(k) == c]].mean(0) for c in CL}
def resid(ks, P): return P - np.stack([cmp_[cat(k)] for k in ks])
rtr = resid(TR, ptr); sdp = rtr.std(0)
bh_tr = np.array([BH[K5[k]] for k in TR]); bh_tr = bh_tr - np.array([np.mean([BH[K5[j]] for j in TR if cat(j) == cat(k)]) for k in TR])
sgn = np.sign(np.corrcoef(rtr[:, 0], bh_tr)[0, 1]); Vt[0] *= sgn; ptr[:, 0] *= sgn; cmp_ = {c: v * np.array([sgn, 1]) for c, v in cmp_.items()}; rtr[:, 0] *= sgn
Ytr = rtr / sdp; Yte = resid(TE, pc_raw(TE)) / sdp
EV = (St ** 2 / (St ** 2).sum()).round(3).tolist()
print(f'질감 PCA 설명 분산 {EV[:4]} · PC1 적재 {np.round(Vt[0], 2).tolist()} · PC2 적재 {np.round(Vt[1], 2).tolist()} · PC1과 band_hi(train) 상관 {abs(np.corrcoef(rtr[:, 0], bh_tr)[0, 1]):.3f}', flush=True)
# ---- 검증 떼기 (범주마다 15 %)
val = np.zeros(len(TR), bool)
for c in CL:
    ii = np.array([i for i, k in enumerate(TR) if cat(k) == c]); val[rng.choice(ii, max(1, int(round(len(ii) * a.val_frac))), replace=False)] = True
fit_i, val_i = np.where(~val)[0], np.where(val)[0]
print(f'train {len(TR)} (학습 {len(fit_i)} · 검증 {len(val_i)}) · test {len(TE)} · 범주 {len(CL)}', flush=True)
# ---- 고정 인코더 출력
raw = lambda s, keys: torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in keys]))
si = SUBS.index(a.sub) if a.sub in SUBS else 0
@torch.no_grad()
def feats(keys):
    o = []
    for i in range(0, len(keys), 32):
        x = raw(a.sub, keys[i:i + 32]).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16): o.append(model.encode_frames(x, torch.full((len(x),), si, dtype=torch.long, device=dev)).float().mean(1).half())
    return torch.cat(o)                                   # [n, 257, 1024]
Xtr, Xte = feats(TR), feats(TE); del model; torch.cuda.empty_cache()
print(f'인코더 출력 {tuple(Xtr.shape)} · {time.time() - t0:.0f}s', flush=True)
def colcorr(P, T):
    P = P - P.mean(0); T = T - T.mean(0); return (P * T).sum(0) / (np.sqrt((P ** 2).sum(0) * (T ** 2).sum(0)) + 1e-12)
def prep(rows):   # 정규화 통계는 학습 행으로만
    # 선형 경로: 펼친 토큰(263k)에 거는 Linear는 학습 행이 펼치는 공간 안에서만 움직이므로(기울기가 그 안에 있음), 그 공간 좌표(학습 행 수 차원)로 바꿔 같은 Linear를 건다 — Adam 보폭 문제를 피함
    f = Xtr[rows].float().flatten(1); mf = f.mean(0, keepdim=True); f -= mf
    ev, U = torch.linalg.eigh((f @ f.t()).double()); keep = ev > ev.max() * 1e-8; U, ev = U[:, keep].float(), ev[keep].float()
    B = (f.t() @ U) / ev.sqrt()                        # [D, r] 정규 직교 기저
    z = f @ B; sz = z.std(); del f
    t = Xtr[rows].float().mean(1); mt = t.mean(0, keepdim=True); st = t.std(0, keepdim=True) + 1e-4
    return lambda X: ((X.float().flatten(1) - mf) @ B / sz, ((X.float().mean(1) - mt) / st))
class Head(nn.Module):
    def __init__(self, D, d=1024, h=256, lin=True, mlp=True):
        super().__init__(); self.lin = nn.Linear(D, 2) if lin else None
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, 2)) if mlp else None
        if self.lin is not None: nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
    def forward(self, xf, xt):
        o = 0
        if self.lin is not None: o = o + self.lin(xf.float())
        if self.mlp is not None: o = o + self.mlp(xt)
        return o
def train_head(rows, cfg, steps=None, evalX=None, evalY=None):
    tf = prep(rows); xf, xt = tf(Xtr[rows]); y = torch.as_tensor(Ytr[rows], dtype=torch.float32, device=dev)
    ex = tf(evalX) if evalX is not None else None
    net = Head(xf.shape[1], lin=cfg['lin'], mlp=cfg['mlp']).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']); n = len(rows); bs = 128
    best, best_s, best_state, MAX, PAT, step, hist = -9, 0, None, steps or 3000, 400, 0, []
    while step < MAX:
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, bs):
            j = perm[b:b + bs]; loss = ((net(xf[j], xt[j]) - y[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if ex is not None and step % 25 == 0:
                with torch.no_grad(): P = net(*ex).cpu().numpy()
                r = colcorr(P, evalY); hist.append((step, float(r[0]), float(r[1])))
                if r[0] > best: best, best_s, best_state = float(r[0]), step, {k: v.clone() for k, v in net.state_dict().items()}
            if step >= MAX or (steps is None and step - best_s > PAT and step > 500): break
        if steps is None and step - best_s > PAT and step > 500: break
    if best_state is not None and steps is None: net.load_state_dict(best_state)
    return net, tf, best, best_s, hist
@torch.no_grad()
def predict(net, tf, X): return net(*tf(X)).cpu().numpy()
def ridge(rows, lam, Xq):
    tf = prep(rows); xf, _ = tf(Xtr[rows]); q, _ = tf(Xq); xf = xf.float(); q = q.float()
    K = xf @ xf.t(); Kq = q @ xf.t(); y = torch.as_tensor(Ytr[rows], dtype=torch.float64, device=dev)
    W = torch.linalg.solve(K.double() + lam * K.diagonal().mean().double() * torch.eye(len(K), device=dev, dtype=torch.float64), y)
    return (Kq.double() @ W).cpu().numpy()
PERM = np.stack([rng.permutation(len(TE)) for _ in range(a.nperm)])
def test_score(P):
    r = colcorr(P, Yte); out = {}
    for d, nm in enumerate(('PC1', 'PC2')):
        nul = np.array([np.corrcoef(P[pp, d], Yte[:, d])[0, 1] for pp in PERM]); out[nm] = [round(float(r[d]), 4), round(float(((nul >= r[d]).sum() + 1) / (a.nperm + 1)), 5)]
    return out
Yv = Ytr[val_i]; R = {'sub': a.sub, 'explained_var': EV, 'pc1_load': np.round(Vt[0], 3).tolist(), 'pc2_load': np.round(Vt[1], 3).tolist(), 'n_fit': len(fit_i), 'n_val': len(val_i), 'configs': {}}
CFGS = {f'head_wd{wd:g}': dict(lin=True, mlp=True, lr=3e-4, wd=wd) for wd in (0.01, 0.1, 1.0)}
CFGS.update({f'mlp_wd{wd:g}': dict(lin=False, mlp=True, lr=3e-4, wd=wd) for wd in (0.1,)})
for name, cfg in CFGS.items():
    net, tf, bv, bs_, hist = train_head(fit_i, cfg, evalX=Xtr[val_i], evalY=Yv); Pv = predict(net, tf, Xtr[val_i]); rv = colcorr(Pv, Yv)
    R['configs'][name] = {'val_PC1': round(float(rv[0]), 4), 'val_PC2': round(float(rv[1]), 4), 'best_step': bs_, 'test_ref': test_score(predict(net, tf, Xte))}
    print(f'{name}: 검증 PC1 {rv[0]:.3f} · PC2 {rv[1]:.3f} (스텝 {bs_}) | test(참고) {R["configs"][name]["test_ref"]} · {time.time() - t0:.0f}s', flush=True)
for lam in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
    Pv = ridge(fit_i, lam, Xtr[val_i]); rv = colcorr(Pv, Yv)
    R['configs'][f'ridge_lam{lam:g}'] = {'val_PC1': round(float(rv[0]), 4), 'val_PC2': round(float(rv[1]), 4), 'lam': lam, 'test_ref': test_score(ridge(fit_i, lam, Xte))}
    print(f'ridge λ {lam:g}: 검증 PC1 {rv[0]:.3f} · PC2 {rv[1]:.3f} | test(참고) {R["configs"][f"ridge_lam{lam:g}"]["test_ref"]}', flush=True)
best = max(R['configs'], key=lambda n: R['configs'][n]['val_PC1']); R['chosen'] = best; allr = np.arange(len(TR))
if best.startswith('ridge'): P = ridge(allr, R['configs'][best]['lam'], Xte)
else:
    torch.manual_seed(0); st = max(100, int(R['configs'][best]['best_step'] * len(TR) / len(fit_i)))
    net, tf, *_ = train_head(allr, CFGS[best], steps=st); P = predict(net, tf, Xte); R['final_steps'] = st
R['test'] = test_score(P); R['chosen_kind'] = 'linear_only(ridge)' if best.startswith('ridge') else 'head(linear+MLP)' if CFGS[best]['lin'] else 'head(MLP only)'
print(f'고른 설정 {best} ({R["chosen_kind"]}) → train 1,300 다시 학습 → test PC1 {R["test"]["PC1"]} · PC2 {R["test"]["PC2"]}', flush=True)
s = a.sub.replace('sub-000', 's0'); np.savez(f'{a.out}/head_{s}.npz', uid=np.array([uid(k) for k in TE]), key=np.array(TE), pred=P, true=Yte, Vt=Vt[:2], sdt=sdt, sdp=sdp,
                                               cm=np.stack([cmp_[c] for c in CL]), cl=np.array(CL))
BASE = json.load(open(a.base_prompt, encoding='utf-8')); out = {}
for k, v in zip(TE, P[:, 0]):
    u = uid(k); out[u] = {'plus': a.plus.format(an=BASE[u]), 'minus': a.minus.format(an=BASE[u]), 'alpha': round(a.k * float(np.clip(v, -2, 2)), 4), 'z': round(float(v), 4)}
json.dump(out, open(f'{a.out}/steer_{s}.json', 'w'), indent=1)
al = np.array([v['alpha'] for v in out.values()]); R['alpha_range'] = [round(float(al.min()), 3), round(float(al.max()), 3)]
json.dump(R, open(f'{a.out}/head_{s}.json', 'w'), ensure_ascii=False, indent=1)
print(f'V10_HEAD_DONE {s} α {R["alpha_range"]} · {time.time() - t0:.0f}s', flush=True)
