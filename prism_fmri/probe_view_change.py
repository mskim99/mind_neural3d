# 전제 확인 (11-09, 로컬 CPU): fMRI가 "같은 물체 안에서 프레임마다 달라지는 정보"(변하는 정보)를 읽나 — V를 '변하는 정보'로 정의할 수 있는지
#  정답: prep_view_targets.py의 실루엣 [cx, cy, w, h, area] (fMRI 프레임 2~7 ↔ 영상 프레임 round(19.2k))
#  이중 중심화 (변하는 부분만): 물체 평균(6프레임)을 빼고, 다시 프레임 번호마다 train 물체 평균을 뺌 → 물체 고유값 · 모든 물체 공통 회전 효과 제거
#  특징: 원 fMRI 프레임(피질 픽셀, 프레임 하나씩)도 같은 이중 중심화 → 표준화(train) → 선형 커널 ridge (alpha는 train LOO, 열마다)
#  평가: train 물체로 맞추고 test 104 물체 × 6프레임에서 Pearson r (열마다). 우연 기준 = test 물체 안에서 프레임 순서를 섞은 순열 (--perm번) → p
#  비교(참고): 물체 평균 값(고정 정보)을 물체 평균 fMRI로 맞춘 r — 예전 Scale r 0.656이 고정 정보 때문인지 보려고
# 출력 <out> json, 로그에 PROBE_VIEW 줄, 끝에 PROBE_VIEW_DONE
# 사용: python -I probe_view_change.py --data_root <fMRI-Shape> --sil <view_sil.npz> --out <json> [--subs sub-0001,...] [--perm 200]
import os, json, argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True); p.add_argument('--sil', required=True); p.add_argument('--out', required=True)
p.add_argument('--subs', default='sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008'); p.add_argument('--perm', type=int, default=200)
a = p.parse_args()
NAMES = ['cx', 'cy', 'w', 'h', 'area']; FRAMES = list(range(2, 8)); ALPHA_REL = [10.0 ** k for k in np.arange(-3, 3.01, 0.5)]
rng = np.random.default_rng(0)
z = np.load(a.sil); KEYS = list(z['keys']); SIL = z['sil'].astype(np.float64)
lst = lambda sp: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
TR, TE = set(lst('train')), set(lst('test'))


def kridge(K, tr, te, Y):
    """선형 커널 ridge, 커널은 train 쪽으로 중심화, alpha는 열마다 LOO."""
    Kt = K[np.ix_(tr, tr)]; rm = Kt.mean(0); am = Kt.mean(); Kc = Kt - rm[None] - rm[:, None] + am
    s, U = np.linalg.eigh(Kc); s = np.clip(s, 0, None); ym = Y[tr].mean(0); Yc = Y[tr] - ym; UtY = U.T @ Yc; U2 = U ** 2; best = []
    for j in range(Y.shape[1]):
        e = []
        for al in [r * s.mean() for r in ALPHA_REL]:
            h = s / (s + al); d = np.clip(1 - U2 @ h - 1 / len(tr), 1e-6, None)
            e.append(((((Yc[:, j] - U @ (h * UtY[:, j])) / d) ** 2).sum(), al))
        best.append(min(e)[1])
    C = U @ (UtY / (s[:, None] + np.array(best)[None]))
    Ke = K[np.ix_(te, tr)]; return (Ke - rm[None] - Ke.mean(1, keepdims=True) + am) @ C + ym


def r_cols(P, Y): return [float(np.corrcoef(P[:, j], Y[:, j])[0, 1]) for j in range(Y.shape[1])]


R = {'note': '이중 중심화(물체 평균 · 프레임별 train 평균 제거) 후 프레임 간 변화, test 104 × 6', 'names': NAMES}
for sub in a.subs.split(','):
    keys = [k for k in KEYS if os.path.exists(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')) and not np.isnan(SIL[KEYS.index(k)]).any()]
    if not keys: print(sub, '데이터 없음'); continue
    ld = lambda k: np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy'), mmap_mode='r')[FRAMES].astype(np.float32)
    m = np.zeros((256, 256), bool)
    for k in keys[:30]: m |= (ld(k) != 0).any(0)
    X = np.stack([ld(k)[:, m] for k in keys])                                   # [N, 6, D]
    Y = np.stack([SIL[KEYS.index(k)] for k in keys])                            # [N, 6, 5]
    istr = np.array([k in TR for k in keys]); iste = np.array([k in TE for k in keys])
    # 고정 정보 (참고): 물체 평균 fMRI → 물체 평균 실루엣
    Xo, Yo = X.mean(1), Y.mean(1)
    Zo = (Xo - Xo[istr].mean(0)) / (Xo[istr].std(0) + 1e-6); Ko = (Zo @ Zo.T).astype(np.float64) / Zo.shape[1]
    r_static = r_cols(kridge(Ko, np.where(istr)[0], np.where(iste)[0], Yo), Yo[iste])
    # 변하는 정보: 이중 중심화
    Xd = X - X.mean(1, keepdims=True); Xd = Xd - Xd[istr].mean(0, keepdims=True)
    Yd = Y - Y.mean(1, keepdims=True); Yd = Yd - Yd[istr].mean(0, keepdims=True)
    F = Xd.reshape(-1, Xd.shape[-1]); T = Yd.reshape(-1, 5); smp_tr = np.repeat(istr, 6); smp_te = np.repeat(iste, 6)
    Zs = (F - F[smp_tr].mean(0)) / (F[smp_tr].std(0) + 1e-6); K = (Zs @ Zs.T).astype(np.float64) / Zs.shape[1]; del Zs, F
    P = kridge(K, np.where(smp_tr)[0], np.where(smp_te)[0], T); Tt = T[smp_te]
    r = r_cols(P, Tt); nte = int(iste.sum()); Pr = P.reshape(nte, 6, 5); null = []
    for _ in range(a.perm):
        q = np.stack([Pr[i, rng.permutation(6)] for i in range(nte)]).reshape(-1, 5); null.append(r_cols(q, Tt))
    null = np.array(null); pv = [float(((null[:, j] >= r[j]).sum() + 1) / (a.perm + 1)) for j in range(5)]
    R[sub] = {'r_change': np.round(r, 3).tolist(), 'p_change': pv, 'r_static_ref': np.round(r_static, 3).tolist(), 'n_train': int(istr.sum()), 'n_test': nte, 'dim': int(m.sum())}
    print(f'PROBE_VIEW {sub} | 변하는 정보 r ' + ' '.join(f'{n} {v:+.3f}(p {q:.3f})' for n, v, q in zip(NAMES, r, pv)) +
          ' | 고정 정보(참고) r ' + ' '.join(f'{n} {v:+.3f}' for n, v in zip(NAMES, r_static)), flush=True)
    json.dump(R, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('PROBE_VIEW_DONE', flush=True)
