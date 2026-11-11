# S · V · P 분리 판정 (11-10, 설계 `.보고 자료/261112_9차보고/PRISM_분리/분리·텍스트_설계.md` 1절) — numpy · torch만 (CPU)
#  입력: 피험자마다 test 추론 components.npz (infer_prism.py, 여러 개) + V 정답 npz (build_view_targets.py, 'shape')
#  판정은 학습에 쓴 판별기가 아니라 여기서 새로 학습하는 탐침으로 (GRL 실패 교훈: 판별기는 속여도 새 탐침은 범주를 읽음)
#  ① 범주 누수   : S · V · P → 범주 (선형 로지스틱 + MLP), 물체 단위 5겹 (같은 물체의 다른 피험자는 같은 겹)
#  ② 피험자 누수 : S · V · P → 피험자 (같은 탐침, 같은 겹) — 피험자 2명 이상일 때만
#  ③ 형상 누수   : S · V · P → V 정답 [cy, w, h, area] ridge R² (같은 겹)
#  ④ 독립성      : 갈래 쌍 선형 CKA · RBF CKA (모든 표본) + 같은 피험자 안에서 물체를 섞은 기준선 (편향 추정량이라 독립이어도 0 아님)
#  (11-11) components.npz에 tok_pool(infer_prism.py --tok_pool 1)이 있으면 예측 토큰 요약도 같은 탐침 → 'tok' 줄
#  ⑤ 유효 랭크   : 갈래별 · 노름 1로 맞춰 붙인 것 (Roy & Vetterli 2007)
#  탐침 정규화 세기는 몇 값 중 가장 높은 test 겹 값을 냄 → 누수를 크게 잡는 쪽(보수적). 우연 = 가장 많은 라벨 비율
# 사용: python sep_eval.py --comp <a/components.npz> [<b/components.npz> …] --vt <view_targets_shape.npz> --out <json> [--name 이름]
import os, json, argparse, numpy as np, torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--comp', nargs='+', required=True); p.add_argument('--vt', default=''); p.add_argument('--out', required=True)
p.add_argument('--name', default=''); p.add_argument('--folds', type=int, default=5); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)

# ---- 읽기: 피험자마다 components.npz, 물체 이름으로 묶음
X = {k: [] for k in ('sem', 'var', 'perc')}; cls, subj, obj = [], [], []; TOK = []
for si, f in enumerate(a.comp):
    z = np.load(f, allow_pickle=True); n = len(z['name'])
    for k in X: X[k].append(z[k].astype(np.float64))
    if 'tok_pool' in z.files: TOK.append(z['tok_pool'].astype(np.float32).reshape(n, -1))
    cls += list(z['cls']); subj += [si] * n; obj += [str(s) for s in z['name']]
X = {k: np.concatenate(v) for k, v in X.items()}; cls = np.array(cls); subj = np.array(subj); obj = np.array(obj)
if len(TOK) == len(a.comp):   # 11-11 검토 반영: 갈래 평균 벡터만 보면 디코더가 쓰는 토큰의 피험자 정보를 놓침 → 예측 토큰 요약도 탐침 (고정 무작위 투영 1024차원, 라벨 안 씀)
    T_ = np.concatenate(TOK); X['tok'] = (T_ @ np.random.RandomState(1).randn(T_.shape[1], 1024).astype(np.float32) / np.sqrt(T_.shape[1])).astype(np.float64)
keep = cls >= 0; X = {k: v[keep] for k, v in X.items()}; cls, subj, obj = cls[keep], subj[keep], obj[keep]
N = len(cls); uo = np.unique(obj); perm = rng.permutation(len(uo)); fold_of = {o: i % a.folds for i, o in enumerate(uo[perm])}
fold = np.array([fold_of[o] for o in obj])
print(f'[sep] {a.name} 표본 {N} (피험자 {len(a.comp)} × 물체 {len(uo)}), 겹 {a.folds}', flush=True)

VT = None
if a.vt and os.path.exists(a.vt):
    zv = np.load(a.vt); m = {str(k).split('/')[-1]: (zv['shape'][i].mean(0), bool(zv['shape_ok'][i])) for i, k in enumerate(zv['keys'])}
    ok = np.array([m.get(o.split('/')[-1], (None, False))[1] for o in obj])
    VT = (np.stack([m[o.split('/')[-1]][0] if ok[i] else np.zeros(4) for i, o in enumerate(obj)]), ok)


def zs(tr, te):
    mu, sd = tr.mean(0), tr.std(0) + 1e-6; return (tr - mu) / sd, (te - mu) / sd


def logreg(Xtr, ytr, Xte, nc, wd):
    W = torch.zeros(Xtr.shape[1], nc, dtype=torch.float64, requires_grad=True); b = torch.zeros(nc, dtype=torch.float64, requires_grad=True)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn='strong_wolfe')
    def clo():
        opt.zero_grad(); l = F.cross_entropy(xt @ W + b, yt) + wd * (W ** 2).sum(); l.backward(); return l
    opt.step(clo)
    return (torch.from_numpy(Xte) @ W + b).argmax(-1).numpy()


def mlp(Xtr, ytr, Xte, nc, wd):
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(256, nc)).double()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd); xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    for _ in range(300):
        opt.zero_grad(); F.cross_entropy(net(xt), yt).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(torch.from_numpy(Xte)).argmax(-1).numpy()


def probe(Xf, y, kind):
    nc = int(y.max()) + 1; best = 0.0
    for wd in ((1e-4, 1e-3, 1e-2) if kind == 'lin' else (1e-2, 1e-1)):
        pred = np.zeros_like(y)
        for k in range(a.folds):
            tr, te = fold != k, fold == k; xa, xb = zs(Xf[tr], Xf[te])
            pred[te] = (logreg if kind == 'lin' else mlp)(xa, y[tr], xb, nc, wd)
        best = max(best, float((pred == y).mean()))
    return best


def ridge_r2(Xf, Y, ok):
    best = -9.0
    for al in (1.0, 10.0, 100.0, 1000.0):
        P = np.zeros_like(Y)
        for k in range(a.folds):
            tr, te = (fold != k) & ok, (fold == k) & ok; xa, xb = zs(Xf[tr], Xf[te]); mu = Y[tr].mean(0)
            Wt = np.linalg.solve(xa.T @ xa + al * np.eye(xa.shape[1]), xa.T @ (Y[tr] - mu)); P[te] = xb @ Wt + mu
        r2 = [1 - ((P[ok, j] - Y[ok, j]) ** 2).sum() / ((Y[ok, j] - Y[ok, j].mean()) ** 2).sum() for j in range(Y.shape[1])]
        best = max(best, float(np.mean(r2)))
    return best


def cka_lin(A, B):
    A = A - A.mean(0); B = B - B.mean(0); return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def cka_rbf(A, B):
    def g(Z):
        sq = (Z ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None] - 2 * Z @ Z.T, 0); med = np.median(d2[~np.eye(len(Z), dtype=bool)]); K = np.exp(-d2 / max(med, 1e-9))
        H = np.eye(len(Z)) - 1.0 / len(Z); return H @ K @ H
    ka, kb = g(A), g(B); return float((ka * kb).sum() / np.sqrt((ka * ka).sum() * (kb * kb).sum()))


def erank(Z):
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False); q = s / s.sum(); q = q[q > 0]; return float(np.exp(-(q * np.log(q)).sum()))


R = {'name': a.name, 'n': N, 'n_subj': len(a.comp), 'n_obj': int(len(uo)), 'chance_cls': float(np.bincount(cls).max() / N)}
# 피험자 평균 빼기 (11-11): 피험자별 ViT · S · V라 갈래마다 피험자 평균 쏠림이 있어 원 특징은 피험자 탐침이 늘 1.0
#  → 원 특징('')과 피험자 평균을 뺀 특징('_c') 둘 다 판정. 분리 비교는 '_c'로 (Slot3D 11-05: 평균만 빼면 피험자 정보 사라짐)
def judge(X, T):
    Q = {}
    for k in [k for k in ('sem', 'var', 'perc', 'tok') if k in X]:
        Q[f'cls_lin_{k}'] = probe(X[k], cls, 'lin'); Q[f'cls_mlp_{k}'] = probe(X[k], cls, 'mlp')
        if len(a.comp) > 1:
            Q[f'subj_lin_{k}'] = probe(X[k], subj, 'lin'); Q[f'subj_mlp_{k}'] = probe(X[k], subj, 'mlp')
        if VT is not None: Q[f'shape_r2_{k}'] = ridge_r2(X[k], VT[0], VT[1])
        Q[f'erank_{k}'] = erank(X[k])
        print(f'SEP|{k} 범주 선형 {Q[f"cls_lin_{k}"]:.3f} MLP {Q[f"cls_mlp_{k}"]:.3f}'
              + (f' | 피험자 선형 {Q[f"subj_lin_{k}"]:.3f} MLP {Q[f"subj_mlp_{k}"]:.3f}' if len(a.comp) > 1 else '')
              + (f' | 형상 R² {Q[f"shape_r2_{k}"]:.3f}' if VT is not None else '') + f' | 유효 랭크 {Q[f"erank_{k}"]:.1f}', flush=True)
    if len(a.comp) > 1: Q['chance_subj'] = float(np.bincount(subj).max() / N)
    for x, y in (('sem', 'var'), ('sem', 'perc'), ('var', 'perc')):
        Q[f'cka_lin_{x}_{y}'] = cka_lin(X[x], X[y]); Q[f'cka_rbf_{x}_{y}'] = cka_rbf(X[x], X[y])
    # CKA 기준선: 같은 피험자 안에서 물체만 섞은 짝 (편향 추정량이라 독립이어도 0이 아님 → 이 값과 비교), 10번 평균
    def shuffled_idx(rs):
        idx = np.arange(N)
        for s in np.unique(subj):
            m = np.where(subj == s)[0]; idx[m] = m[rs.permutation(len(m))]
        return idx
    for x, y in (('sem', 'var'), ('sem', 'perc'), ('var', 'perc')):
        bl, br = [], []
        for t in range(10):
            ix = shuffled_idx(np.random.RandomState(100 + t)); bl.append(cka_lin(X[x], X[y][ix])); br.append(cka_rbf(X[x], X[y][ix]))
        Q[f'cka_lin_{x}_{y}_base'] = float(np.mean(bl)); Q[f'cka_rbf_{x}_{y}_base'] = float(np.mean(br))
    print('SEP|CKA 섞은 기준선 선형 S-V {:.3f} S-P {:.3f} V-P {:.3f} | RBF S-V {:.3f} S-P {:.3f} V-P {:.3f}'.format(
        Q['cka_lin_sem_var_base'], Q['cka_lin_sem_perc_base'], Q['cka_lin_var_perc_base'], Q['cka_rbf_sem_var_base'], Q['cka_rbf_sem_perc_base'], Q['cka_rbf_var_perc_base']), flush=True)
    nb = {k: X[k] / np.linalg.norm(X[k] - X[k].mean(0)) for k in ('sem', 'var', 'perc')}
    Q['erank_sum'] = sum(erank(nb[k]) for k in nb); Q['erank_joint'] = erank(np.hstack([nb[k] for k in ('sem', 'var', 'perc')]))
    print('SEP|CKA 선형 S-V {:.3f} S-P {:.3f} V-P {:.3f} | RBF S-V {:.3f} S-P {:.3f} V-P {:.3f} | 유효 랭크 합 {:.1f} vs 붙임 {:.1f} | 우연 범주 {:.3f}'.format(
        Q['cka_lin_sem_var'], Q['cka_lin_sem_perc'], Q['cka_lin_var_perc'], Q['cka_rbf_sem_var'], Q['cka_rbf_sem_perc'], Q['cka_rbf_var_perc'],
        Q['erank_sum'], Q['erank_joint'], R['chance_cls']) + (f' 피험자 {Q["chance_subj"]:.3f}' if 'chance_subj' in Q else ''), flush=True)
    return {k + T: v for k, v in Q.items()}


Xc = {k: v.copy() for k, v in X.items()}
for k in Xc:
    for s_ in np.unique(subj): Xc[k][subj == s_] -= Xc[k][subj == s_].mean(0)
for T, XX in (('', X), ('_c', Xc)):
    print(f'SEP|=== {"원 특징" if T == "" else "피험자 평균 뺀 특징"}', flush=True); R.update(judge(XX, T))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(R, open(a.out, 'w'), indent=1)
print('SEP_DONE', a.out, flush=True)
