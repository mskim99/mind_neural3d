# PRISM 색 선형 프로브 (11-06, MinD-3D++ venv, 실행 위치 $W/run) — fMRI에서 자극 물체의 색이 읽히는가
#  정답: 자극 영상 프레임 0 (<tgt>/frames/<uid>/0.png)의 물체 영역(min RGB < 235, dino 정답 토큰과 같은 기준) 평균 R · G · B · 밝기 · 채도(max−min)
#  특징 (피험자마다, test fMRI): sem_vec(768) · var_vec(768) · perc_vec(256) · tok(예측 조건 토큰 평균 1024)
#   + 고정 MinD-3D++ ViT 출력 (우리 학습 이전): vit_cls(프레임 평균 CLS 1024) · vit_patch(프레임 · 패치 평균 1024) · vit_cls6(프레임 6개 CLS 이어붙임 6144)
#   → vit가 범주 기준선보다 높으면 fMRI(인코더)에 색이 있는데 우리 모델이 버린 것, 아니면 인코더 단계부터 색이 없음
#  비교: 범주 기준선 = train 물체의 같은 범주 평균 색 (범주만 알면 나오는 값) / 상한 = 정답 DINOv2 토큰 평균(1024)으로 같은 프로브
#  프로브: 표준화 + ridge, test 104개 5겹 교차검증 (alpha는 학습 겹 안에서 다시 5겹으로 고름). 지표 Pearson r, R²
#  추가: train→test (train fMRI 특징으로 맞추고 test에 적용. 모델이 train에 맞춰져 있어 낙관적일 수 있음)
# 출력 <out>/probe_color.json, 로그에 PROBE 줄
# 사용: python probe_color.py --repo <MinD-3D> --data_root <fMRI-Shape> --tgt <shape_targets> --ckpt <model_final.pt> --out <폴더> [--subs a,b]
import os, sys, json, argparse, numpy as np, torch
from PIL import Image
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True); p.add_argument('--subs', default=''); p.add_argument('--bs', type=int, default=8)
a = p.parse_args()
FRAMES = list(range(2, 8)); TN = ['R', 'G', 'B', '밝기', '채도']; ALPHAS = [10.0 ** k for k in range(-2, 6)]


def color(uid):
    rgb = np.asarray(Image.open(f'{a.tgt}/frames/{uid}/0.png').convert('RGB')).astype(np.float32)
    m = rgb.min(2) < 235; px = rgb[m] if m.sum() > 50 else rgb.reshape(-1, 3)
    return np.array([*px.mean(0), px.mean(), (px.max(1) - px.min(1)).mean()], np.float32)


def ridge_fit(X, Y, al):
    mu, sd = X.mean(0), X.std(0) + 1e-6; ym = Y.mean(0); Z = (X - mu) / sd
    W = np.linalg.solve(Z.T @ Z + al * np.eye(Z.shape[1]), Z.T @ (Y - ym)) if Z.shape[0] > Z.shape[1] else \
        Z.T @ np.linalg.solve(Z @ Z.T + al * np.eye(Z.shape[0]), Y - ym)            # 표본 < 차원이면 쌍대 형태
    return lambda Xn: ((Xn - mu) / sd) @ W + ym


def folds(n, k=5, seed=0):
    idx = np.random.RandomState(seed).permutation(n); return [idx[i::k] for i in range(k)]


def fit_best(X, Y):                       # 학습 겹 안에서 alpha 고르기
    err = []
    for al in ALPHAS:
        e = 0
        for va in folds(len(X), 5, 1):
            tr = np.setdiff1d(np.arange(len(X)), va); e += ((ridge_fit(X[tr], Y[tr], al)(X[va]) - Y[va]) ** 2).sum()
        err.append(e)
    return ridge_fit(X, Y, ALPHAS[int(np.argmin(err))])


def cv_pred(X, Y):
    P = np.zeros_like(Y)
    for te in folds(len(X)):
        tr = np.setdiff1d(np.arange(len(X)), te); P[te] = fit_best(X[tr], Y[tr])(X[te])
    return P


def score(P, Y):
    r = [float(np.corrcoef(P[:, j], Y[:, j])[0, 1]) if P[:, j].std() > 0 else 0.0 for j in range(Y.shape[1])]
    r2 = [float(1 - ((P[:, j] - Y[:, j]) ** 2).sum() / ((Y[:, j] - Y[:, j].mean()) ** 2).sum()) for j in range(Y.shape[1])]
    return {'r': dict(zip(TN, r)), 'R2': dict(zip(TN, r2))}


ids = lambda sp: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
tr_ids = [k for k in ids('train') if os.path.exists(f'{a.tgt}/frames/{k.split("/")[1]}/0.png')]; te_ids = ids('test')
Ytr = np.stack([color(k.split('/')[1]) for k in tr_ids]); Yte = np.stack([color(k.split('/')[1]) for k in te_ids])
print(f'정답 색 | train {len(tr_ids)} test {len(te_ids)} | test 채도 평균 {Yte[:, 4].mean():.1f} 범위 {Yte[:, 4].min():.1f}~{Yte[:, 4].max():.1f}', flush=True)
R = {'targets': TN, 'n_train': len(tr_ids), 'n_test': len(te_ids)}
# 범주 기준선: train 같은 범주 평균 (없으면 train 전체 평균)
cat = lambda k: k.split('/')[0]
cm = {c: Ytr[[cat(k) == c for k in tr_ids]].mean(0) for c in set(map(cat, tr_ids))}
R['category_mean'] = score(np.stack([cm.get(cat(k), Ytr.mean(0)) for k in te_ids]), Yte)
# 상한: 정답 DINOv2 토큰 평균
G = np.stack([np.load(f'{a.tgt}/dino/{k.split("/")[1]}.npy').astype(np.float32).mean(0) for k in te_ids])
R['gt_dino_cv'] = score(cv_pred(G, Yte), Yte)
show = lambda name, s: print(f"PROBE {name:<22} r " + ' '.join(f"{t} {s['r'][t]:+.2f}" for t in TN) + " | R² " + ' '.join(f"{s['R2'][t]:+.2f}" for t in TN), flush=True)
show('범주 기준선', R['category_mean']); show('상한 정답 DINO (cv)', R['gt_dino_cv'])

ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']
model = build_from_ckpt(ck, a.repo).cuda().eval()


def feats(sub, keys):
    si = SUBS.index(sub); F = {n: [] for n in NAMES}
    for i in range(0, len(keys), a.bs):
        ks = keys[i:i + a.bs]
        x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
        fi = torch.tensor(FRAMES).expand(len(ks), -1).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            sj = torch.full((len(ks),), si, dtype=torch.long, device='cuda')
            o = model(x, fi, sj); e = model.encode_frames(x, sj).float()                  # e [B, 6, 257, 1024], 0번 = CLS
        for n, v in (('sem', o['sem_vec']), ('var', o['var_vec']), ('perc', o['perc_vec']), ('tok', o['cond_tokens'].mean(1)),
                     ('vit_cls', e[:, :, 0].mean(1)), ('vit_patch', e[:, :, 1:].mean((1, 2))), ('vit_cls6', e[:, :, 0].flatten(1))):
            F[n].append(v.float().cpu().numpy())
    return {n: np.concatenate(v) for n, v in F.items()}


NAMES = ('vit_cls', 'vit_patch', 'vit_cls6', 'sem', 'var', 'perc', 'tok')
R['subs'] = {}
for sub in (a.subs.split(',') if a.subs else SUBS):
    Fte = feats(sub, te_ids); Ftr = feats(sub, tr_ids); R['subs'][sub] = {}
    for n in NAMES:
        s = score(cv_pred(Fte[n], Yte), Yte); s['train_to_test'] = score(fit_best(Ftr[n], Ytr)(Fte[n]), Yte); R['subs'][sub][n] = s
        show(f'{sub} {n} (cv)', s); show(f'{sub} {n} (train→test)', s['train_to_test'])
if R['subs']:                                  # 피험자 평균
    R['mean'] = {n: {m: {t: float(np.mean([R['subs'][s][n][m][t] for s in R['subs']])) for t in TN} for m in ('r', 'R2')} for n in NAMES}
    for n in R['mean']: show(f'평균 {n} (cv)', R['mean'][n])
os.makedirs(a.out, exist_ok=True); json.dump(R, open(f'{a.out}/probe_color.json', 'w'), indent=1, ensure_ascii=False)
print('PROBE_DONE', flush=True)
