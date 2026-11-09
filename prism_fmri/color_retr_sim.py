# 색 유도 검색 모의 (11-09, 로컬 CPU): fMRI 원 신호의 약한 색 정보로 "어떤 train 물체 외형을 가져올지" 고르면 정답 색에 가까워지나
#  배경: 외형(SLat) = 실제 train 토큰 하나(검색)일 때 렌더 색 ≈ 그 train 물체 자극 색 (정답 토큰 렌더 색 일치 r 0.87~0.91) → 렌더 없이 고른 물체의 자극 색으로 잼
#  색 예측: 피험자마다 원 신호(프레임 2~7 평균 피질 픽셀) → 선형 커널 ridge(alpha LOO) → test 104 색 코드 (정규채도 · rg_n · by_n), probe_color_raw.py와 같은 특징 · 정답
#  고르기 (모두 test 정답 범주 안 train 물체에서 — 범주는 고정하고 색만 봄):
#    rand   = 범주 안 무작위 (200번 평균) = 범주만 알 때
#    probe  = 색 예측과 가장 가까운 물체 (색 코드는 train 표준편차로 나눠 거리)
#    oracle = 정답 색과 가장 가까운 물체 (검색 방식의 상한)
#    tok    = --slat_map이 있으면 그 피험자 실제 검색 결과(토큰 기준)도 같이
#  지표: 104개에 걸친 Pearson r(고른 물체 자극 색, test 자극 색) 색마다 + 범주 평균을 뺀 잔차 r + 거기서 test 물체 면적 · 밝기까지 뺀 부분 r (← 판정, 형태 · 음영과 얽힌 몫 제외). probe는 범주 안 예측 섞기 순열 p
# 사용: python color_retr_sim.py --data_root <fMRI-Shape> --npz data/stim_color.npz --subs sub-0001,...,sub-0008 --out <json> [--slat_map sub-0001=<slat_map.json>]
import os, json, argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True); p.add_argument('--npz', required=True); p.add_argument('--subs', required=True)
p.add_argument('--out', required=True); p.add_argument('--perm', type=int, default=200); p.add_argument('--slat_map', action='append', default=[])
a = p.parse_args()
FRAMES = list(range(2, 8)); CN = ['정규채도', 'rg_n', 'by_n']; COL = [3, 4, 5]
rng = np.random.default_rng(0)
lst = lambda sp: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
z = np.load(a.npz); CODE = dict(zip(list(z['keys']), z['Y'][:, COL].astype(np.float64)))
tr = [k for k in lst('train') if k in CODE and not np.isnan(CODE[k]).any()]; te = [k for k in lst('test') if k in CODE]
cat = lambda k: k.split('/')[0]
YT, YE = np.stack([CODE[k] for k in tr]), np.stack([CODE[k] for k in te])
SD = YT.std(0); TC = np.array([cat(k) for k in tr]); EC = np.array([cat(k) for k in te])
CM = {c: YT[TC == c].mean(0) for c in set(TC)}                     # train 범주 평균 (잔차 기준)
RES = lambda Y: Y - np.stack([CM[c] for c in EC])


def rr(X, Y): return [float(np.corrcoef(X[:, j], Y[:, j])[0, 1]) for j in range(3)]


AB = np.stack([z['Y'][list(z['keys']).index(k), :2] for k in te]).astype(np.float64)   # test 물체 면적 · 밝기
AB1 = np.c_[AB - AB.mean(0), np.ones(len(AB))]
PART = lambda Y: Y - AB1 @ np.linalg.lstsq(AB1, Y, rcond=None)[0]                     # 범주 잔차에서 면적 · 밝기를 더 뺌 (probe_color_raw의 부분 잔차와 같은 뜻)


def score(P):                                                      # P [104, 3] 고른 물체 자극 색
    return {'r': rr(P, YE), 'res_r': rr(RES(P), RES(YE)), 'part_r': rr(PART(RES(P)), PART(RES(YE)))}


def pick(pred):                                                    # 범주 안에서 pred와 가장 가까운 train 물체 색
    out = []
    for c, q in zip(EC, pred):
        idx = np.where(TC == c)[0]; d = (((YT[idx] - q) / SD) ** 2).sum(1); out.append(YT[idx[d.argmin()]])
    return np.stack(out)


def ridge_pred(X, Y, tr_i, te_i):                                  # 선형 커널 ridge, alpha는 LOO로 열마다
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6); K = Z @ Z.T / Z.shape[1]
    Kt = K[np.ix_(tr_i, tr_i)]; rm = Kt.mean(0); am = Kt.mean(); Kc = Kt - rm[None] - rm[:, None] + am
    s, U = np.linalg.eigh(Kc); s = np.clip(s, 0, None); ym = Y.mean(0); Yc = Y - ym; UtY = U.T @ Yc; best = []
    for j in range(Y.shape[1]):
        errs = []
        for al in [r * s.mean() for r in 10.0 ** np.arange(-3, 3.01, 0.5)]:
            h = s / (s + al); d = np.clip(1 - (U ** 2) @ h - 1 / len(tr_i), 1e-6, None)
            errs.append(((((Yc[:, j] - U @ (h * UtY[:, j])) / d) ** 2).sum(), al))
        best.append(min(errs)[1])
    C = U @ (UtY / (s[:, None] + np.array(best)[None]))
    Ke = K[np.ix_(te_i, tr_i)]; Kec = Ke - rm[None] - Ke.mean(1, keepdims=True) + am
    return Kec @ C + ym


R = {'note': '범주 = test 정답 범주 고정, 색 코드 = 자극 영상 48프레임', 'n_test': len(te), 'n_train': len(tr), 'cols': CN}
base = {'rand': [score(np.stack([YT[rng.choice(np.where(TC == c)[0])] for c in EC])) for _ in range(200)], 'oracle': score(pick(YE))}
R['rand'] = {k: np.mean([b[k] for b in base['rand']], 0).round(3).tolist() for k in ('r', 'res_r', 'part_r')}
R['oracle'] = {k: np.round(v, 3).tolist() for k, v in base['oracle'].items()}
print('범주 무작위', R['rand'], '| 정답 색으로 고르기(상한)', R['oracle'], flush=True)
for sm in a.slat_map:
    sub, f = sm.split('=', 1); M = json.load(open(f, encoding='utf-8')); keep = [i for i, k in enumerate(te) if k.split('/')[1] in M]
    P = np.stack([CODE[M[te[i].split('/')[1]]['train']] for i in keep]); Yk = YE[keep]; ECk = EC[keep]
    resk = lambda Y: Y - np.stack([CM[c] for c in ECk])
    R[f'tok_{sub}'] = {'r': np.round(rr(P, Yk), 3).tolist(), 'res_r': np.round(rr(resk(P), resk(Yk)), 3).tolist(), 'n': len(keep)}
    print(f'토큰 검색 {sub}', R[f'tok_{sub}'], flush=True)
allk = tr + te; tr_i = np.arange(len(tr)); te_i = np.arange(len(tr), len(allk)); Yall = np.concatenate([YT, YE])
for sub in a.subs.split(','):
    if not all(os.path.exists(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')) for k in allk):
        print(sub, '파일 빠짐 → 건너뜀', flush=True); continue
    mn = np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy'), mmap_mode='r')[FRAMES].astype(np.float32).mean(0).ravel() for k in allk])
    X = mn[:, (mn != 0).any(0)]; del mn
    pred = ridge_pred(X, Yall[tr_i], tr_i, te_i); sc = score(pick(pred))
    pr = {k: [] for k in ('r', 'res_r', 'part_r')}                             # 순열: 범주 안에서 예측을 섞음
    for _ in range(a.perm):
        q = pred.copy()
        for c in set(EC):
            i = np.where(EC == c)[0]; q[i] = pred[rng.permutation(i)]
        s2 = score(pick(q)); [pr[k].append(s2[k]) for k in pr]
    pv = {k: [float(((np.array(pr[k])[:, j] >= sc[k][j]).sum() + 1) / (a.perm + 1)) for j in range(3)] for k in pr}
    R[sub] = {'probe_direct_r': np.round(rr(pred, YE), 3).tolist(), **{k: np.round(v, 3).tolist() for k, v in sc.items()}, 'p': pv, 'dim': int(X.shape[1])}
    print(sub, R[sub], flush=True)
    json.dump(R, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
json.dump(R, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('COLOR_RETR_SIM_DONE', flush=True)
