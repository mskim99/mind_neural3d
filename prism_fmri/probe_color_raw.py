# fMRI 원 신호 색 프로브 (11-08, CPU, 로컬에서도 돎) — 인코더를 거치기 전 fMRI에 범주를 넘는 색 정보가 있나
#  11-06 probe_color.py는 인코더(MinD-3D++ ViT) 이후 특징만 봄 → 범주 기준선보다 낮음. 여기서는 원 신호를 직접 보고, 범주 평균을 뺀 잔차 색을 맞힘
#  정답 (물체마다): 자극 영상(stimuli.zip mp4, 192프레임 = 한 바퀴)의 4프레임마다 48장, 물체 영역(min RGB < 235, 2px 깎음) 기준
#   면적(물체 픽셀 비율) · 밝기(픽셀 평균) · 채도(max−min, 음영 영향 받음) — 형태 · 음영과 얽힌 값, 비교용
#   정규채도(픽셀마다 (max−min)/max 평균) · rg_n = 3(R−G)/(R+G+B) · by_n = 3((R+G)/2−B)/(R+G+B) — 음영에 덜 민감한 색 지표 (판정용)
#  특징 (피험자마다): raw = 프레임 2~7 평균 평면 지도의 피질 픽셀 (0이 아닌 픽셀, 약 3만) / t6 = 프레임 6장 각각 8×8 평균 풀링 이어붙임 (6144)
#  프로브: 특징 표준화(전체 표본, 정답 안 씀) + 선형 커널 ridge, 커널은 학습 쪽 평균으로 중심화, alpha는 학습 쪽 LOO(절편 포함)로 정답 열마다
#  평가: A = train → test 104 (PRISM과 같은 분할) / B = 전체 10겹 교차검증
#  지표 (Pearson r):
#   범주 기준선(학습 쪽 같은 범주 평균) · 직접 · 합(범주 평균 + 예측 잔차)
#   잔차 r = (정답 − 범주 평균)을 맞춘 r. 색 지표 3개는 면적 · 밝기 잔차를 학습 쪽 선형회귀로 더 뺀 "부분 잔차"도 냄 (← 판정 기준)
#   우연 기준: 범주 안에서 정답을 섞는 순열 --perm번 (같은 순열을 모든 피험자 · A · B에 씀) → 열마다 p, 부분 잔차 3개는 최댓값 순열로 묶음 보정 p
#   양성 대조: 범주 정확도(범주 사이) · 면적 잔차 r(범주 안) — 면적은 읽히는데 색은 안 읽히면 "색이 없다"로 읽을 수 있음
#   6명 평균 = 피험자별 예측 평균 (순열도 같은 방식으로 평균해 자기 우연 기준)
#  한계: run · session 정보를 쓰지 않음(순열이 run 구조를 보존하지 않음)
# 출력 <out>/probe_color_raw.json, <out>/stim_color.npz(정답 캐시), 로그에 STIM · PROBE 줄, 끝에 PROBE_RAW_DONE
# 사용: python probe_color_raw.py --data_root <fMRI-Shape> --out <폴더> [--subs sub-0001,...] [--jobs 8] [--perm 200] [--feats raw,t6]
import os, json, zipfile, argparse, tempfile, numpy as np
from concurrent.futures import ThreadPoolExecutor
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True); p.add_argument('--out', required=True); p.add_argument('--subs', default='')
p.add_argument('--jobs', type=int, default=8); p.add_argument('--perm', type=int, default=200); p.add_argument('--feats', default='raw,t6')
p.add_argument('--limit', type=int, default=0)                       # 연기 시험용: 범주마다 이 개수만
a = p.parse_args()
FRAMES = list(range(2, 8)); VF = list(range(0, 192, 4)); KOUT = 10
TN = ['면적', '밝기', '채도', '정규채도', 'rg_n', 'by_n']; COL = [3, 4, 5]; COV = [0, 1]
RN = TN + [f'{TN[c]}|면적·밝기 제거' for c in COL]                    # 잔차 열 9개 (뒤 3개 = 부분 잔차, 판정 기준)
PART = list(range(len(TN), len(RN)))
ALPHA_REL = [10.0 ** k for k in np.arange(-3, 3.01, 0.5)]


def stim_color(key):
    import cv2
    z = zipfile.ZipFile(os.path.join(a.data_root, 'stimuli.zip')); ker = np.ones((5, 5), np.uint8)
    fd, tmp = tempfile.mkstemp(suffix='.mp4'); os.close(fd)
    try:
        open(tmp, 'wb').write(z.read(f'stimuli/{key}.mp4')); c = cv2.VideoCapture(tmp); px, area, i = [], [], 0
        while i <= max(VF):
            ok, f = c.read()
            if not ok: break
            if i in VF:
                m = cv2.erode((f.min(2) < 235).astype(np.uint8), ker).astype(bool); area.append(m.mean())
                if m.sum() > 50: px.append(f[m][:, ::-1].astype(np.float32))          # BGR → RGB
            i += 1
        c.release()
    finally:
        os.remove(tmp)
    if not px: return np.full(len(TN), np.nan, np.float32)
    x = np.concatenate(px); R, G, B = x.mean(0); mx, mn = x.max(1), x.min(1); S = R + G + B + 1e-6
    return np.array([np.mean(area), x.mean(), (mx - mn).mean(), ((mx - mn) / np.maximum(mx, 1)).mean(), 3 * (R - G) / S, 3 * ((R + G) / 2 - B) / S], np.float32)


def load_sub(sub, keys):
    X = np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy'), mmap_mode='r')[FRAMES].astype(np.float32) for k in keys])   # [n, 6, 256, 256]
    F = {}
    if 'raw' in a.feats:
        mn = X.mean(1).reshape(len(X), -1); m = (mn != 0).any(0); F['raw'] = mn[:, m]
    if 't6' in a.feats:
        F['t6'] = X.reshape(len(X), 6, 32, 8, 32, 8).mean((3, 5)).reshape(len(X), -1)
    return F


def kernel(F):
    Z = (F - F.mean(0)) / (F.std(0) + 1e-6); return (Z @ Z.T).astype(np.float64) / Z.shape[1]


class KRidge:                                                       # 학습 쪽 커널로 맞추고, alpha는 정답 열마다 LOO로 (절편 지렛값 1/n 포함)
    def __init__(self, K, tr):
        Kt = K[np.ix_(tr, tr)]; self.rm = Kt.mean(0); self.am = Kt.mean(); self.tr = tr
        Kc = Kt - self.rm[None] - self.rm[:, None] + self.am; s, U = np.linalg.eigh(Kc); self.s = np.clip(s, 0, None); self.U = U
        self.al = [r * self.s.mean() for r in ALPHA_REL]; U2 = U ** 2
        self.H = [(self.s / (self.s + al), np.clip(1 - U2 @ (self.s / (self.s + al)) - 1.0 / len(tr), 1e-6, None)) for al in self.al]

    def fit(self, Y):
        ym = Y.mean(0); Yc = Y - ym; UtY = self.U.T @ Yc
        E = np.stack([(((Yc - self.U @ (h[:, None] * UtY)) / d[:, None]) ** 2).sum(0) for h, d in self.H])   # [alpha, 열]
        best = np.array(self.al)[E.argmin(0)]
        self.C = self.U @ (UtY / (self.s[:, None] + best[None])); self.ym = ym; return self

    def predict(self, K, te):
        Kq = K[np.ix_(te, self.tr)]; Kq = Kq - Kq.mean(1, keepdims=True) - self.rm[None] + self.am; return Kq @ self.C + self.ym


def r_cols(P, Y):
    P = P - P.mean(0); Y = Y - Y.mean(0); d = np.sqrt((P ** 2).sum(0) * (Y ** 2).sum(0))
    return np.where(d > 1e-12, (P * Y).sum(0) / np.maximum(d, 1e-12), 0.0)


def cat_mean(Y, cats, tr, idx):
    g = Y[tr].mean(0); cm = {c: Y[tr][cats[tr] == c].mean(0) for c in np.unique(cats[tr])}
    return np.stack([cm.get(cats[i], g) for i in idx])


def resid_targets(Y, cats, tr, te):                                 # 범주 잔차 6열 + 면적 · 밝기 잔차를 학습 쪽 회귀로 뺀 색 부분 잔차 3열
    Rtr = Y[tr] - cat_mean(Y, cats, tr, tr); Rte = Y[te] - cat_mean(Y, cats, tr, te)
    Ctr = np.c_[np.ones(len(tr)), Rtr[:, COV]]; Cte = np.c_[np.ones(len(te)), Rte[:, COV]]
    beta = np.linalg.lstsq(Ctr, Rtr[:, COL], rcond=None)[0]
    return np.c_[Rtr, Rtr[:, COL] - Ctr @ beta], np.c_[Rte, Rte[:, COL] - Cte @ beta]


def run_split(K, Y, cats, splits, krs, direct=True):              # → 직접 예측, 잔차 예측(9열), 잔차 정답(9열), 범주 기준선
    n = len(Y); Pd, Cb = np.full_like(Y, np.nan), np.full_like(Y, np.nan); Pr, Tr = np.full((n, len(RN)), np.nan), np.full((n, len(RN)), np.nan)
    for (tr, te), kr in zip(splits, krs):
        Rtr, Rte = resid_targets(Y, cats, tr, te)
        if direct: Pd[te] = kr.fit(Y[tr]).predict(K, te)
        Pr[te] = kr.fit(Rtr).predict(K, te); Tr[te] = Rte; Cb[te] = cat_mean(Y, cats, tr, te)
    return Pd, Pr, Tr, Cb


def summarize(Y, out, idx):
    Pd, Pr, Tr, Cb = (x[idx] for x in out); Y = Y[idx]
    return {'category': r_cols(Cb, Y).tolist(), 'direct': r_cols(Pd, Y).tolist(), 'resid': r_cols(Pr, Tr).tolist(), 'combined': r_cols(Cb + Pr[:, :len(TN)], Y).tolist()}


def null_stats(obs, null):                                          # obs [9], null [P, 9] → 95% · 열마다 p · 부분 잔차 묶음 보정 p
    null = np.asarray(null); P = len(null); mx = null[:, PART].max(1)
    return {'null95': np.percentile(null, 95, 0).tolist(), 'p': ((1 + (null >= obs[None]).sum(0)) / (1 + P)).tolist(),
            'p_fwe_part': {RN[j]: float((1 + (mx >= obs[j]).sum()) / (1 + P)) for j in PART}, 'max_null95_part': float(np.percentile(mx, 95))}


ann = lambda sp: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
tr_k, te_k = ann('train'), ann('test')
if a.limit:
    lim = lambda L: [k for c in sorted({k.split('/')[0] for k in L}) for k in [x for x in L if x.startswith(c)][:a.limit]]
    tr_k, te_k = lim(tr_k), lim(te_k)
keys = tr_k + te_k
os.makedirs(a.out, exist_ok=True); cache = f'{a.out}/stim_color.npz'
if os.path.exists(cache) and list(np.load(cache)['keys']) == keys and list(np.load(cache)['names']) == TN:
    Y = np.load(cache)['Y']
else:
    with ThreadPoolExecutor(a.jobs) as ex: Y = np.stack(list(ex.map(stim_color, keys)))
    np.savez(cache, Y=Y, keys=np.array(keys), names=np.array(TN))
bad = np.isnan(Y).any(1); is_te = np.array([False] * len(tr_k) + [True] * len(te_k))
if bad.any(): print(f'물체 영역 없음 {bad.sum()}개 → 제외: ' + ' '.join(np.array(keys)[bad]), flush=True)
keep = np.nonzero(~bad)[0]; keys = [keys[i] for i in keep]; Y = Y[keep].astype(np.float64); is_te = is_te[keep]
n = len(keys); cats = np.array([k.split('/')[0] for k in keys]); A_tr, A_te = np.nonzero(~is_te)[0], np.nonzero(is_te)[0]
ch, nch = Y[:, 2], Y[:, 3]; allidx = np.arange(n)
R = {'targets': TN, 'resid_targets': RN, 'n': n, 'n_train': len(A_tr), 'n_test': len(A_te), 'excluded': int(bad.sum()), 'video_frames': VF, 'perm': a.perm,
     'stim': {'chroma_mean': float(ch.mean()), 'chroma_pct': {q: float(np.percentile(ch, q)) for q in (10, 25, 50, 75, 90)},
              'gray_frac_chroma_lt10': float((ch < 10).mean()), 'gray_frac_chroma_lt20': float((ch < 20).mean()),
              'normchroma_mean': float(nch.mean()), 'gray_frac_normchroma_lt005': float((nch < 0.05).mean()), 'gray_frac_normchroma_lt010': float((nch < 0.10).mean()),
              'by_category': {c: {'채도': float(ch[cats == c].mean()), '정규채도': float(nch[cats == c].mean()), '회색비율(정규채도<0.1)': float((nch[cats == c] < 0.1).mean())} for c in np.unique(cats)},
              'category_R2': (1 - ((Y - cat_mean(Y, cats, allidx, allidx)) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)).tolist()}}
st = R['stim']
print(f"STIM n {n} (제외 {bad.sum()}) | 채도 평균 {st['chroma_mean']:.1f} 중앙값 {st['chroma_pct'][50]:.1f} (10~90% {st['chroma_pct'][10]:.1f}~{st['chroma_pct'][90]:.1f}) · 채도<10 {st['gray_frac_chroma_lt10']:.2f} · <20 {st['gray_frac_chroma_lt20']:.2f}", flush=True)
print(f"STIM 정규채도 평균 {st['normchroma_mean']:.3f} · <0.05 {st['gray_frac_normchroma_lt005']:.2f} · <0.10 {st['gray_frac_normchroma_lt010']:.2f}", flush=True)
print('STIM 범주별 정규채도(회색 비율) ' + ' '.join(f"{c}:{v['정규채도']:.2f}({v['회색비율(정규채도<0.1)']:.2f})" for c, v in st['by_category'].items()), flush=True)
print('STIM 범주가 설명하는 분산 R² ' + ' '.join(f'{t} {v:.2f}' for t, v in zip(TN, st['category_R2'])), flush=True)

B_perm = np.random.RandomState(0).permutation(n); B_splits = [(np.setdiff1d(allidx, B_perm[i::KOUT]), np.sort(B_perm[i::KOUT])) for i in range(KOUT)]
A_splits = [(A_tr, A_te)]
rs = np.random.RandomState(1); PERMS = []
for t in range(a.perm):
    pi = allidx.copy()
    for c in np.unique(cats):
        m = np.nonzero(cats == c)[0]; pi[m] = m[rs.permutation(len(m))]
    PERMS.append(pi)
fmt = lambda names, v: ' '.join(f'{t} {x:+.2f}' for t, x in zip(names, v))
CL = np.unique(cats); OH = (cats[:, None] == CL[None]).astype(np.float64)


def cat_acc(K, splits, krs, idx):
    P = np.full_like(OH, np.nan)
    for (tr, te), kr in zip(splits, krs): P[te] = kr.fit(OH[tr]).predict(K, te)
    return float((P[idx].argmax(1) == OH[idx].argmax(1)).mean())


def report(tag, sA, sB):
    for nm, s in (('A', sA), ('B', sB)):
        print(f"PROBE {tag} {nm} 범주 기준선 {fmt(TN, s['category'])} | 직접 {fmt(TN, s['direct'])} | 합 {fmt(TN, s['combined'])}", flush=True)
        print(f"PROBE {tag} {nm} 잔차 r {fmt(RN, s['resid'])}", flush=True)
        if 'null' in s:
            print(f"PROBE {tag} {nm} 잔차 우연95% {fmt(RN, s['null']['null95'])} | p " + ' '.join(f'{t} {x:.3f}' for t, x in zip(RN, s['null']['p'])), flush=True)
            print(f"PROBE {tag} {nm} 색 부분 잔차 묶음 보정 p " + ' '.join(f'{t} {x:.3f}' for t, x in s['null']['p_fwe_part'].items()) + f" (최댓값 우연95% {s['null']['max_null95_part']:+.2f})", flush=True)


SUBS = a.subs.split(',') if a.subs else sorted(d for d in os.listdir(a.data_root) if d.startswith('sub-'))
R['subs'] = {}; ens = {}
for sub in SUBS:
    F = load_sub(sub, keys); R['subs'][sub] = {}
    print(f'{sub} 특징 ' + ' '.join(f'{k} {v.shape[1]}' for k, v in F.items()), flush=True)
    for fn in list(F):
        K = kernel(F.pop(fn))
        krA = [KRidge(K, tr) for tr, _ in A_splits]; krB = [KRidge(K, tr) for tr, _ in B_splits]
        oA, oB = run_split(K, Y, cats, A_splits, krA), run_split(K, Y, cats, B_splits, krB)
        sA, sB = summarize(Y, oA, A_te), summarize(Y, oB, allidx)
        sA['cat_acc'], sB['cat_acc'] = cat_acc(K, A_splits, krA, A_te), cat_acc(K, B_splits, krB, allidx)
        e = ens.setdefault(fn, {'oA': [], 'oB': [], 'nA': np.zeros((a.perm, n, len(RN))), 'nB': np.zeros((a.perm, n, len(RN))), 'tA': [None] * a.perm, 'tB': [None] * a.perm})
        e['oA'].append(oA); e['oB'].append(oB)
        if a.perm:
            nullA, nullB = [], []
            for t, pi in enumerate(PERMS):
                Yp = Y[pi]; _, PrA, TrA, _ = run_split(K, Yp, cats, A_splits, krA, False); _, PrB, TrB, _ = run_split(K, Yp, cats, B_splits, krB, False)
                nullA.append(r_cols(PrA[A_te], TrA[A_te])); nullB.append(r_cols(PrB, TrB))
                e['nA'][t] += PrA; e['nB'][t] += PrB; e['tA'][t], e['tB'][t] = TrA, TrB
            sA['null'] = null_stats(np.array(sA['resid']), nullA); sB['null'] = null_stats(np.array(sB['resid']), nullB)
        R['subs'][sub][fn] = {'A_train_to_test': sA, 'B_cv10': sB}
        print(f"PROBE {sub} {fn} 양성 대조: 범주 정확도 A {sA['cat_acc']:.3f} · B {sB['cat_acc']:.3f} (우연 {1 / len(CL):.3f})", flush=True)
        report(f'{sub} {fn}', sA, sB)
        json.dump(R, open(f'{a.out}/probe_color_raw.json', 'w'), indent=1, ensure_ascii=False)
if len(SUBS) > 1:
    R['ensemble'] = {}
    for fn, e in ens.items():
        avg = lambda L: tuple(np.mean([x[i] for x in L], 0) for i in range(4))
        sA, sB = summarize(Y, avg(e['oA']), A_te), summarize(Y, avg(e['oB']), allidx)
        if a.perm:
            S = len(e['oA'])
            sA['null'] = null_stats(np.array(sA['resid']), [r_cols(e['nA'][t][A_te] / S, e['tA'][t][A_te]) for t in range(a.perm)])
            sB['null'] = null_stats(np.array(sB['resid']), [r_cols(e['nB'][t] / S, e['tB'][t]) for t in range(a.perm)])
        R['ensemble'][fn] = {'A_train_to_test': sA, 'B_cv10': sB}
        report(f'{len(SUBS)}명 평균 {fn}', sA, sB)
json.dump(R, open(f'{a.out}/probe_color_raw.json', 'w'), indent=1, ensure_ascii=False)
print('PROBE_RAW_DONE', flush=True)
