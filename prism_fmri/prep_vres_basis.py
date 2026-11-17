# -*- coding: utf-8 -*-
# Variational 개선 임시 실험 준비 + 전제 확인 (11-17, 추가 논문 개선안 — 사용자 승인: 디코더 유지 + 더하기, A와 B 같이)
#  ① 정답 · 기저: 학습 물체(train에서 범주마다 끝 n개를 뗀 것, --holdout)의 GT 토큰(<tgt>/dino/<uid>.npy, 자극 프레임 0의 DINOv2 = TRELLIS 조건 토큰)으로
#     범주 평균 토큰 [13, 1374, 1024]과 잔차(GT − 범주 평균)의 PCA 기저 k개를 구함 → 물체마다 계수(학습 물체 표준편차로 나눔)를 Variational 정답으로
#     출력 npz는 train_prism.py --view_tgt 형식(keys · shape [N, 6, k] · shape_ok)에 res_mean · res_U · res_sd를 같이 넣음 (--v_res_basis로 그대로 씀)
#  ② 전제 확인 (A): test 물체에서 L_tok(MSE + 1 − cos)을 비교 — 정답 범주의 평균 토큰 / train 전체 평균 토큰 / 범주 평균 + PCA k개(상한) / 모델 예측(--pred)
#     + 같은 범주 8개 안에서 정답 물체 고르기(예측 토큰과 GT 토큰의 cos 최대), 범주 안 우연 = 1/8
# 사용: python prep_vres_basis.py --tgt $TGT --data_root $DR --holdout 8 --k 32 --out $TGT/vres_basis_k32.npz [--pred 이름=토큰폴더 ...]
#  토큰 폴더 = infer_prism.py 출력의 <피험자>_keep/tokens (물체마다 <uid>.npy [1374, 1024])
import argparse, os, json, time
import numpy as np, torch
p = argparse.ArgumentParser()
p.add_argument('--tgt', required=True); p.add_argument('--data_root', required=True); p.add_argument('--holdout', type=int, default=8)
p.add_argument('--k', type=int, default=32); p.add_argument('--out', required=True); p.add_argument('--pred', nargs='*', default=[])
a = p.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'; t0 = time.time()
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
tr_all, te = lst('train'), lst('test')
by = {}
for k in tr_all: by.setdefault(k.split('/')[0], []).append(k)
H = {k for v in by.values() for k in v[-a.holdout:]} if a.holdout > 0 else set()
tr = [k for k in tr_all if k not in H]; SYN = sorted(by)                      # 범주 순서 = train_prism.py SYN (sorted)
ok = lambda k: os.path.exists(f'{a.tgt}/dino/{k.split("/")[1]}.npy'); tr, te, ho = [k for k in tr if ok(k)], [k for k in te if ok(k)], [k for k in H if ok(k)]
tok = lambda k: torch.from_numpy(np.load(f'{a.tgt}/dino/{k.split("/")[1]}.npy').astype(np.float16)).to(dev)
X = torch.stack([tok(k) for k in tr]); N, T, C = X.shape; D = T * C; cl = torch.tensor([SYN.index(k.split('/')[0]) for k in tr], device=dev)
print(f'학습 물체 {N} · 검증 {len(ho)} · test {len(te)} · 토큰 {T}×{C}', flush=True)
mean = torch.stack([X[cl == c].float().mean(0) for c in range(len(SYN))])         # [13, T, C]
gmean = X.float().mean(0)
R = (X.float() - mean[cl]).view(N, D)                                             # 잔차
G = R @ R.t(); ev, V = torch.linalg.eigh(G.double()); ev, V = ev.flip(0)[:a.k].float(), V.flip(1)[:, :a.k].float()
U = (R.t() @ V) / ev.clamp_min(1e-6).sqrt()                                       # [D, k] 단위 기저
expl = float(ev.sum() / torch.trace(G)); del G
coef = lambda x, c: (x.float().view(len(x), D) - mean[c].view(len(x), D)) @ U     # [n, k]
ctr = coef(X, cl); sd = ctr.std(0).clamp_min(1e-6)
allk = tr + ho + te; Z = []
for i in range(0, len(allk), 64):
    ks = allk[i:i + 64]; xx = torch.stack([tok(k) for k in ks]); cc = torch.tensor([SYN.index(k.split('/')[0]) for k in ks], device=dev)
    Z.append((coef(xx, cc) / sd).cpu())
Z = torch.cat(Z).numpy().astype(np.float32)
np.savez(a.out, keys=np.array(allk), shape=np.repeat(Z[:, None], 6, 1), shape_ok=np.ones(len(allk), bool),
         res_mean=mean.half().cpu().numpy(), res_U=U.t().contiguous().view(a.k, T, C).half().cpu().numpy(), res_sd=sd.cpu().numpy(), classes=np.array(SYN))
print(f'저장 {a.out} | 잔차 분산 중 PCA {a.k}개 몫 {expl:.3f} | {time.time() - t0:.0f}s', flush=True)

# ---------- ② 전제 확인 (test)
def ltok(pr, gt):                                                                  # 학습 손실과 같은 식 (물체마다)
    return (((pr - gt) ** 2).mean((1, 2)) + (1 - torch.nn.functional.cosine_similarity(pr, gt, dim=-1)).mean(1))
Y = torch.stack([tok(k) for k in te]).float(); cy = torch.tensor([SYN.index(k.split('/')[0]) for k in te], device=dev)
res = {'n_test': len(te), 'k': a.k, 'pca_explained_train': round(expl, 4)}
res['L_tok 정답 범주 평균 토큰'] = float(ltok(mean[cy], Y).mean())
res['L_tok train 전체 평균 토큰'] = float(ltok(gmean[None].expand_as(Y), Y).mean())
cte = coef(Y, cy); res[f'L_tok 범주 평균 + PCA {a.k}개 (상한)'] = float(ltok(mean[cy] + (cte @ U.t()).view(-1, T, C), Y).mean())


def within(pred):                                                                  # 같은 범주 test 물체 중 정답 고르기
    pf = torch.nn.functional.normalize(pred.view(len(pred), -1), dim=-1); yf = torch.nn.functional.normalize(Y.view(len(Y), -1), dim=-1); S = pf @ yf.t(); hit = []
    for i in range(len(te)):
        m = (cy == cy[i]).nonzero().squeeze(1); hit.append(int(m[S[i, m].argmax()] == i))
    return float(np.mean(hit)), float(np.mean([1 / int((cy == c).sum()) for c in cy]))
for spec in a.pred:
    name, d = spec.split('=', 1); ks = [k for k in te if os.path.exists(f'{d}/{k.split("/")[1]}.npy')]
    if len(ks) < len(te): print(f'[{name}] 토큰 {len(ks)}/{len(te)}개만 있음 — 있는 것만');
    idx = [te.index(k) for k in ks]; P = torch.stack([torch.from_numpy(np.load(f'{d}/{k.split("/")[1]}.npy')).float().to(dev) for k in ks])
    Yk = Y[idx]; res[f'L_tok 모델 [{name}]'] = float(ltok(P, Yk).mean())
    if len(ks) == len(te): acc, ch = within(P); res[f'같은 범주 안 정답 고르기 [{name}]'] = round(acc, 3); res['같은 범주 안 우연'] = round(ch, 3)
acc, ch = within(mean[cy] + (cte @ U.t()).view(-1, T, C)); res[f'같은 범주 안 정답 고르기 [범주 평균 + PCA {a.k} 상한]'] = round(acc, 3)
out = a.out.replace('.npz', '_premise.json'); json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
for k, v in res.items(): print(f'{k}: {v}')
print('PREMISE_DONE', out)
