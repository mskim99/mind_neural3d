# PRISM 분기 분리 프로브 (11-06, MinD-3D++ venv, 실행 위치 $W/run) — ① 크기 비율이 fMRI로 읽히나 ② sem · var · perc가 실제로 나뉘어 있나
#  정답 (물체마다):
#   범주 (분류) — annotations 경로의 synset
#   scale · cx · cy — 영상 실루엣 값 (<tgt>/targets/<uid>.npz), fMRI 프레임 2~7에 대응하는 영상 프레임 평균
#   크기 비율 ext_x · ext_y · ext_z — 64³ 복셀(<tgt>/voxels, 가장 긴 변 = 1)에서 축별로 차 있는 범위 / 64. ShapeNet은 y가 위
#  특징: vit_cls (고정 MinD-3D++ ViT, 프레임 평균 CLS) · sem · var · perc · tok — probe_color.py와 같음
#  프로브: 표준화 + ridge (alpha는 학습 안 5겹으로 고름), train fMRI로 맞추고 test에 적용 (train→test) + test 5겹 cv
#   범주: one-hot ridge → argmax 정확도. 기준 = 최빈 범주, 우연 = 1/범주 수
#   연속값: Pearson r. 범주 평균을 빼고 남은 부분(잔차)을 다시 맞춘 r도 냄 → 0보다 확실히 크면 범주를 넘는 정보
#  읽는 법: 범주 정확도는 sem이 가장 높고 var · perc는 낮아야 분리됨. 크기 비율 · scale 잔차 r은 var가 가장 높아야 함
# 출력 <out>/probe_split.json, 로그에 PROBE 줄, 끝에 PROBE_DONE
# 사용: python probe_split.py --repo <MinD-3D> --data_root <fMRI-Shape> --tgt <shape_targets> --ckpt <model_final.pt> --out <폴더> [--subs sub-0001]
import os, sys, json, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True); p.add_argument('--subs', default='sub-0001'); p.add_argument('--bs', type=int, default=8)
a = p.parse_args()
FRAMES = list(range(2, 8)); VF = [min(191, round(19.2 * f)) for f in FRAMES]; ALPHAS = [10.0 ** k for k in range(-2, 6)]
CT = ['scale', 'cx', 'cy', 'ext_x', 'ext_y', 'ext_z']
NAMES = ('vit_cls', 'sem', 'var', 'perc', 'tok')


def target(uid):
    t = np.load(f'{a.tgt}/targets/{uid}.npz')
    occ = np.unpackbits(np.load(f'{a.tgt}/voxels/{uid}.npy'))[:64 ** 3].reshape(64, 64, 64)
    ext = [(lambda nz: (nz.max() - nz.min() + 1) / 64 if len(nz) else 0.0)(np.nonzero(occ.any(axis=tuple(j for j in range(3) if j != ax)))[0]) for ax in range(3)]
    return np.array([t['scale'][VF].mean(), t['cx'][VF].mean(), t['cy'][VF].mean(), *ext], np.float32)


def ridge_fit(X, Y, al):
    mu, sd = X.mean(0), X.std(0) + 1e-6; ym = Y.mean(0); Z = (X - mu) / sd
    W = np.linalg.solve(Z.T @ Z + al * np.eye(Z.shape[1]), Z.T @ (Y - ym)) if Z.shape[0] > Z.shape[1] else \
        Z.T @ np.linalg.solve(Z @ Z.T + al * np.eye(Z.shape[0]), Y - ym)
    return lambda Xn: ((Xn - mu) / sd) @ W + ym


def folds(n, k=5, seed=0):
    idx = np.random.RandomState(seed).permutation(n); return [idx[i::k] for i in range(k)]


def fit_best(X, Y):
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


rr = lambda P, Y: [float(np.corrcoef(P[:, j], Y[:, j])[0, 1]) if P[:, j].std() > 0 and Y[:, j].std() > 0 else 0.0 for j in range(Y.shape[1])]
ids = lambda sp: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
ok = lambda k: all(os.path.exists(f'{a.tgt}/{d}/{k.split("/")[1]}.{e}') for d, e in (('targets', 'npz'), ('voxels', 'npy')))
tr_ids = [k for k in ids('train') if ok(k)]; te_ids = [k for k in ids('test') if ok(k)]
cat = lambda k: k.split('/')[0]; CATS = sorted(set(map(cat, tr_ids)))
Ytr = np.stack([target(k.split('/')[1]) for k in tr_ids]); Yte = np.stack([target(k.split('/')[1]) for k in te_ids])
Ctr = np.array([CATS.index(cat(k)) for k in tr_ids]); Cte = np.array([CATS.index(cat(k)) if cat(k) in CATS else -1 for k in te_ids])
OH = lambda c: np.eye(len(CATS), dtype=np.float32)[c]
cm = np.stack([Ytr[Ctr == c].mean(0) for c in range(len(CATS))])                 # train 범주 평균 (잔차 기준)
Rtr, Rte = Ytr - cm[Ctr], Yte - cm[np.maximum(Cte, 0)]
print(f'정답 | train {len(tr_ids)} test {len(te_ids)} 범주 {len(CATS)} | test 표준편차 ' + ' '.join(f'{t} {Yte[:, j].std():.3f}' for j, t in enumerate(CT))
      + ' | 범주 안 표준편차 ' + ' '.join(f'{t} {Rte[:, j].std():.3f}' for j, t in enumerate(CT)), flush=True)
maj = float((Cte == np.bincount(Ctr).argmax()).mean())
R = {'targets': CT, 'cats': CATS, 'n_train': len(tr_ids), 'n_test': len(te_ids), 'cat_majority': maj, 'cat_chance': 1 / len(CATS),
     'category_mean_r': dict(zip(CT, rr(cm[np.maximum(Cte, 0)], Yte)))}
print(f'PROBE 기준선            범주 최빈 {maj:.3f} 우연 {1 / len(CATS):.3f} | 범주평균 r ' + ' '.join(f'{t} {v:+.2f}' for t, v in R['category_mean_r'].items()) + ' | 잔차 r 0', flush=True)

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
            o = model(x, fi, sj); e = model.encode_frames(x, sj).float()
        for n, v in (('vit_cls', e[:, :, 0].mean(1)), ('sem', o['sem_vec']), ('var', o['var_vec']), ('perc', o['perc_vec']), ('tok', o['cond_tokens'].mean(1))):
            F[n].append(v.float().reshape(len(ks), -1).cpu().numpy())
    return {n: np.concatenate(v) for n, v in F.items()}


R['subs'] = {}
for sub in a.subs.split(','):
    Fte = feats(sub, te_ids); Ftr = feats(sub, tr_ids); R['subs'][sub] = {}
    for n in NAMES:
        acc = float((fit_best(Ftr[n], OH(Ctr))(Fte[n]).argmax(1) == Cte).mean())
        acc_cv = float((cv_pred(Fte[n], OH(np.maximum(Cte, 0))).argmax(1) == Cte).mean())
        r = rr(fit_best(Ftr[n], Ytr)(Fte[n]), Yte); res = rr(fit_best(Ftr[n], Rtr)(Fte[n]), Rte); res_cv = rr(cv_pred(Fte[n], Rte), Rte)
        R['subs'][sub][n] = {'cat_acc': acc, 'cat_acc_cv': acc_cv, 'r': dict(zip(CT, r)), 'res_r': dict(zip(CT, res)), 'res_r_cv': dict(zip(CT, res_cv))}
        print(f'PROBE {sub} {n:<8} 범주 {acc:.3f} (cv {acc_cv:.3f}) | r ' + ' '.join(f'{t} {v:+.2f}' for t, v in zip(CT, r))
              + ' | 잔차 r ' + ' '.join(f'{t} {v:+.2f}' for t, v in zip(CT, res)) + ' | 잔차 cv ' + ' '.join(f'{v:+.2f}' for v in res_cv), flush=True)
os.makedirs(a.out, exist_ok=True); json.dump(R, open(f'{a.out}/probe_split.json', 'w'), indent=1, ensure_ascii=False)
print('PROBE_DONE', flush=True)
