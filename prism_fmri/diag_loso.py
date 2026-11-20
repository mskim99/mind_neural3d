# -*- coding: utf-8 -*-
# 11-20 진단 (P-1): 짝 데이터 없이 처음 보는 피험자에게 범주 정보가 옮겨지나 — 학습 없음(커널 릿지 one-hot, λ 0.1 · diag_align.py와 같은 분류기)
#  절차: 7명 중 1명을 빼고, 남은 6명의 train 물체를 합쳐 범주 분류기를 맞춤 → 뺀 1명의 test 104개에 그대로 적용 (뺀 사람의 라벨 · 짝은 안 씀). 7번 반복
#        + 7명 전부로 맞춘 분류기를 --ap 피험자(train 없음)의 test에 적용
#  입력 형식: pixel 평면 지도 피질 화소(프레임 2~7 평균, in_norm) / region 영역 평균(좌우 합침) · regionLR(좌우 따로) / blur2 · blur4 · blur8 가우시안 σ 화소(피질 마스크로 정규화해 밖 화소가 안 섞임)
#  피험자별 맞춤: raw 없음(맞춘 피험자들의 train 평균만 뺌) / ztrain 피험자마다 자기 train의 화소별 평균 · 표준편차 / ztest 새 피험자는 자기 test 104개의 평균 · 표준편차(맞춘 피험자는 train)
#  대조: own 자기 train → 자기 test (ztrain) / one2one 한 사람 → 한 사람 42쌍 (ztrain) / 순열 p (test 라벨을 섞어 10,000번, 정확도가 이 이상일 확률)
#  영역 단위: 영역 하나의 화소만 써서 같은 절차 (ztrain, AP는 ztest)
# 사용 (실행 위치 $W/run): python diag_loso.py --data_root $DR --subs sub-0001,...,sub-0008 --ap sub-0009 --out $W/prism_calib/loso
import os, json, time, argparse, numpy as np, torch
import torch.nn.functional as F
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True); p.add_argument('--subs', required=True); p.add_argument('--ap', default=''); p.add_argument('--out', required=True)
p.add_argument('--min_px', type=int, default=40); p.add_argument('--perm', type=int, default=10000); p.add_argument('--no_region', action='store_true')
p.add_argument('--extra', default=''); p.add_argument('--only_extra', action='store_true'); p.add_argument('--same', default='1,6,7,8')   # 11-20: train도 있는 새 피험자(4번)에 7명 분류기 적용 + 제시 순서 묶음별
a = p.parse_args()
dev = 'cuda'; t0 = time.time(); SUBS = a.subs.split(','); NS = len(SUBS); FRAMES = list(range(2, 8)); LAM = 0.1
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
ex = lambda s, k_: os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy'))
TR = [k_ for k_ in lst('train') if all(ex(s, k_) for s in SUBS)]; TE = [k_ for k_ in lst('test') if all(ex(s, k_) for s in SUBS)]; TE9 = [k_ for k_ in lst('test') if a.ap and ex(a.ap, k_)]
CL = sorted({k_.split('/')[0] for k_ in TR}); NC = len(CL); yv = lambda ks: torch.tensor([CL.index(k_.split('/')[0]) for k_ in ks], device=dev)
ytr, yte, yte9 = yv(TR), yv(TE), yv(TE9); ntr, nte = len(TR), len(TE)
here = os.path.dirname(os.path.abspath(__file__))
lab = torch.as_tensor(np.load(os.path.join(here, 'assets', 'mmp_256_regions.npy'))).long().to(dev); M2 = lab > 0; M = M2.flatten(); labm = lab.flatten()[M]
NAME = {int(k): v for k, v in json.load(open(os.path.join(here, 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items() if int(k) > 0}
REG = {}
for r, n in NAME.items(): REG.setdefault(n.split('_', 1)[-1], []).append(r)
print(f'train {ntr} · test {nte} · AP {a.ap} test {len(TE9)} | 범주 {NC} | 피험자 {NS} | 화소 {int(M.sum())} | 영역 {len(REG)}(좌우 합침) · {len(NAME)}(따로)', flush=True)


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


@torch.no_grad()
def collect(s, keys):                                                      # → [N, 256, 256] 프레임 평균 지도 (half)
    o = []
    for i in range(0, len(keys), 32):
        x = norm_in(torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')).astype(np.float32)[FRAMES] for k_ in keys[i:i + 32]])).to(dev))
        o.append(x.mean(1).half())
    return torch.cat(o)


RAW = {s: {'train': collect(s, TR), 'test': collect(s, TE)} for s in SUBS}
if TE9: RAW[a.ap] = {'test': collect(a.ap, TE9)}
if a.extra: RAW[a.extra] = {'train': collect(a.extra, TR), 'test': collect(a.extra, TE)}; assert all(ex(a.extra, k_) for k_ in TR + TE)
print(f'읽기 끝 {time.time() - t0:.0f}s', flush=True)


def gk(sig):
    r = int(3 * sig + 0.5); x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32); k = torch.exp(-x ** 2 / (2 * sig ** 2)); return (k / k.sum()), r
def blur(x, sig):                                                          # x [N, 256, 256] float → 마스크 정규화 가우시안
    k, r = gk(sig); m = M2.float()[None, None]
    def g(v): v = F.conv2d(v, k.view(1, 1, 1, -1), padding=(0, r)); return F.conv2d(v, k.view(1, 1, -1, 1), padding=(r, 0))
    den = g(m).clamp_min(1e-6); o = []
    for i in range(0, len(x), 256): o.append((g((x[i:i + 256].float() * m[0])[:, None]) / den)[:, 0])
    return torch.cat(o)
RID = torch.zeros(int(lab.max()) + 1, dtype=torch.long, device=dev); RNAMES = sorted(REG)
for i, n in enumerate(RNAMES):
    for r in REG[n]: RID[r] = i
LRIDS = sorted(NAME)
def regmean(x, lr):                                                        # x [N, 256, 256] → [N, 영역 수]
    v = x.flatten(1)[:, M].float()
    if lr: idx = torch.searchsorted(torch.tensor(LRIDS, device=dev), labm); k = len(LRIDS)
    else: idx = RID[labm]; k = len(RNAMES)
    oh = F.one_hot(idx, k).float(); return (v @ oh) / oh.sum(0, keepdim=True)
FMT = {'pixel': lambda x: x.flatten(1)[:, M].float(), 'region': lambda x: regmean(x, False), 'regionLR': lambda x: regmean(x, True),
       'blur2': lambda x: blur(x, 2).flatten(1)[:, M], 'blur4': lambda x: blur(x, 4).flatten(1)[:, M], 'blur8': lambda x: blur(x, 8).flatten(1)[:, M]}
I = torch.eye
def solve(K, T): return torch.linalg.solve(K.double() + (LAM * K.diagonal().mean().double() + 1e-6) * I(len(K), device=dev, dtype=torch.float64), T.double()).float()
def fit(X, y): X = F.normalize(X, dim=-1); return X, solve(X @ X.t(), F.one_hot(y, NC))
def predict(mdl, Z): X, c = mdl; return (F.normalize(Z, dim=-1) @ X.t() @ c).argmax(1)
def zs(x, ref): mu = ref.mean(0, keepdim=True); sd = ref.std(0, keepdim=True).clamp_min(1e-6); return (x - mu) / sd
gperm = torch.Generator(device='cpu').manual_seed(0)
def perm_p(pred, y, n=a.perm):                                             # test 라벨을 섞었을 때 정확도가 이 이상일 확률
    acc = float((pred == y).float().mean()); idx = torch.stack([torch.randperm(len(y), generator=gperm) for _ in range(n)]).to(dev)
    return (float(((pred[None] == y[idx]).float().mean(1) >= acc - 1e-9).float().sum()) + 1) / (n + 1)
def conf(pred, y): return torch.bincount(y * NC + pred, minlength=NC * NC).view(NC, NC).cpu().numpy()   # [정답, 예측]
sh = lambda s: s[-2:].lstrip('0')


def run(feat, tag, full=True):                                             # feat[s] = {'train': [n, d], 'test': [n, d]} (float) → 조건별 결과
    R = {}; CF = {}
    Z = {s: {'train': zs(feat[s]['train'], feat[s]['train']), 'test': zs(feat[s]['test'], feat[s]['train']), 'test_zt': zs(feat[s]['test'], feat[s]['test'])} for s in SUBS}
    if TE9: Z9 = zs(feat[a.ap]['test'], feat[a.ap]['test'])
    for nm in (('raw', 'ztrain', 'ztest') if full else ('ztrain',)):
        accs, ps, pall, cf = {}, {}, [], np.zeros((NC, NC), np.int64)
        for i, s in enumerate(SUBS):
            oth = [t for t in SUBS if t != s]
            if nm == 'raw': mu = torch.cat([feat[t]['train'] for t in oth]).mean(0, keepdim=True); Xtr = torch.cat([feat[t]['train'] for t in oth]) - mu; Xte = feat[s]['test'] - mu
            else: Xtr = torch.cat([Z[t]['train'] for t in oth]); Xte = Z[s]['test' if nm == 'ztrain' else 'test_zt']
            pr = predict(fit(Xtr, ytr.repeat(NS - 1)), Xte); accs[sh(s)] = round(float((pr == yte).float().mean()), 4); cf += conf(pr, yte); pall.append(pr)
            if full: ps[sh(s)] = round(perm_p(pr, yte), 4)
            del Xtr
        r = {'by_sub': accs, 'mean': round(float(np.mean(list(accs.values()))), 4), 'n_over_0.15': int(sum(v > 0.15 for v in accs.values()))}
        if full: r['p_by_sub'] = ps; r['p_pooled'] = round(perm_p(torch.cat(pall), yte.repeat(NS)), 5)
        if TE9 and nm != 'ztrain':
            if nm == 'raw': mu = torch.cat([feat[t]['train'] for t in SUBS]).mean(0, keepdim=True); Xtr = torch.cat([feat[t]['train'] for t in SUBS]) - mu; X9 = feat[a.ap]['test'] - mu
            else: Xtr = torch.cat([Z[t]['train'] for t in SUBS]); X9 = Z9
            pr = predict(fit(Xtr, ytr.repeat(NS)), X9); r['ap'] = round(float((pr == yte9).float().mean()), 4); CF[f'{tag}|{nm}|ap'] = conf(pr, yte9)
            if full: r['ap_p'] = round(perm_p(pr, yte9), 4)
            del Xtr
        R[nm] = r; CF[f'{tag}|{nm}'] = cf
    mdl = [fit(Z[s]['train'], ytr) for s in SUBS]                           # 대조: 자기 / 한 사람 → 한 사람 (ztrain)
    own = [float((predict(mdl[i], Z[s]['test']) == yte).float().mean()) for i, s in enumerate(SUBS)]
    R['own'] = {'by_sub': {sh(s): round(v, 4) for s, v in zip(SUBS, own)}, 'mean': round(float(np.mean(own)), 4)}
    if full: R['one2one'] = {'mean': round(float(np.mean([float((predict(mdl[i], Z[t]['test']) == yte).float().mean()) for i in range(NS) for t in SUBS if t != SUBS[i]])), 4)}
    return R, CF


def run_extra(feat):                                                       # 새 피험자 E (train · test 다 있음): 7명 / 같은 묶음 / 다른 묶음으로 맞춘 분류기를 E의 test에
    E_ = a.extra; R = {}; same = [s for s in SUBS if sh(s) in a.same.split(',')]; GR = {'all7': SUBS, 'same_order': same, 'diff_order': [s for s in SUBS if s not in same]}
    zE = {'raw': None, 'ztrain': zs(feat[E_]['test'], feat[E_]['train']), 'ztest': zs(feat[E_]['test'], feat[E_]['test'])}
    for g, ss in GR.items():
        R[g] = {'subs': [sh(s) for s in ss]}
        for nm in ('raw', 'ztrain', 'ztest'):
            if nm == 'raw': mu = torch.cat([feat[t]['train'] for t in ss]).mean(0, keepdim=True); Xtr = torch.cat([feat[t]['train'] for t in ss]) - mu; Xe = feat[E_]['test'] - mu
            else: Xtr = torch.cat([zs(feat[t]['train'], feat[t]['train']) for t in ss]); Xe = zE[nm]
            pr = predict(fit(Xtr, ytr.repeat(len(ss))), Xe); R[g][nm] = round(float((pr == yte).float().mean()), 4); R[g][nm + '_p'] = round(perm_p(pr, yte), 4); CFS[f'extra|{g}|{nm}|' + CUR[0]] = conf(pr, yte); del Xtr
    R['own'] = round(float((predict(fit(zs(feat[E_]['train'], feat[E_]['train']), ytr), zE['ztrain']) == yte).float().mean()), 4)
    return R


CUR = ['']
OUT = {'chance': round(1 / NC, 4), 'subs': SUBS, 'ap': a.ap, 'n_test': nte, 'n_test_ap': len(TE9), 'classes': CL, 'lam': LAM, 'perm': a.perm, 'formats': {}}; CFS = {}
for f, fn in FMT.items():
    feat = {s: {k_: fn(v) for k_, v in d.items()} for s, d in RAW.items()}
    CUR[0] = f
    if a.only_extra:
        OUT['formats'][f] = {'extra': run_extra(feat)}; print(f'[{f}] {a.extra} ' + json.dumps(OUT['formats'][f]['extra'], ensure_ascii=False), flush=True); del feat; torch.cuda.empty_cache(); continue
    R, CF = run({s: d for s, d in feat.items() if s != a.extra}, f); OUT['formats'][f] = R; CFS.update(CF); OUT['formats'][f]['dim'] = int(feat[SUBS[0]]['test'].shape[1])
    if a.extra: R['extra'] = run_extra(feat)
    print(f'[{f}] ' + ' | '.join(f'{nm} 평균 {R[nm]["mean"]} ({R[nm]["n_over_0.15"]}/{NS}명 > 0.15)' + (f' AP {R[nm]["ap"]}' if 'ap' in R[nm] else '') for nm in ('raw', 'ztrain', 'ztest')) + f' | 자기 {R["own"]["mean"]} · 1→1 {R["one2one"]["mean"]} | {time.time() - t0:.0f}s', flush=True)
    del feat; torch.cuda.empty_cache()
if not a.no_region and not a.only_extra:                                                        # 영역 하나씩 (화소, ztrain · AP는 ztest)
    OUT['regions'] = {}
    for n in RNAMES:
        m = torch.isin(labm, torch.tensor(REG[n], device=dev))
        if int(m.sum()) < a.min_px: continue
        feat = {s: {k_: v.flatten(1)[:, M][:, m].float() for k_, v in d.items()} for s, d in RAW.items() if s != a.extra}
        R, _ = run(feat, n, full=False); r = {'px': int(m.sum()), 'loso': R['ztrain']['mean'], 'loso_by_sub': R['ztrain']['by_sub'], 'own': R['own']['mean']}
        if TE9:
            Xtr = torch.cat([zs(feat[t]['train'], feat[t]['train']) for t in SUBS]); r['ap'] = round(float((predict(fit(Xtr, ytr.repeat(NS)), zs(feat[a.ap]['test'], feat[a.ap]['test'])) == yte9).float().mean()), 4)
        OUT['regions'][n] = r
    top = sorted(OUT['regions'].items(), key=lambda kv: -kv[1]['loso'])[:8]; print('영역 단위 상위:', [(n, r['loso'], r.get('ap'), r['own']) for n, r in top], f'{time.time() - t0:.0f}s', flush=True)
os.makedirs(a.out, exist_ok=True)
OUT['extra'] = a.extra; FN = 'loso_extra' if a.only_extra else 'loso'
json.dump(OUT, open(f'{a.out}/{FN}.json', 'w'), ensure_ascii=False, indent=1); np.savez_compressed(f'{a.out}/{FN}_conf.npz', **CFS)
print('LOSO_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
