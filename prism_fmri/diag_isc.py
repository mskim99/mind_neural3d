# -*- coding: utf-8 -*-
# 11-19 분석: HCP-MMP 영역마다 "같은 물체를 본 피험자 사이의 반응이 얼마나 닮았나" — 학습 없음 (Perceptive 가설을 세우기 전 데이터 확인)
#  입력: 피험자마다 fMRI 프레임 2~7 평균 (in_norm), 영역 픽셀 패턴 (좌우 같은 이름은 합침). 피험자 · 영역마다 train 물체 평균을 뺌
#  영역마다 피험자 쌍(7명 → 21쌍) 평균:
#    obj_top1      test 104개에서, 피험자 A의 물체 패턴으로 피험자 B의 같은 물체 고르기 (패턴 상관, 우연 1/104)
#    obj_top1_wc   같은 범주 안에서 고르기 (우연 ≈ 1/8) — 범주 평균을 뺀 물체 고유 패턴
#    obj_gap       같은 물체 상관 − 다른 물체 상관 (평균)
#    cat_cross     피험자 A의 범주 중심(train 물체 평균 패턴)으로 피험자 B의 test 물체 범주 맞히기 (우연 1/13)
#    cat_within    자기 범주 중심으로 자기 test 물체 범주 맞히기 (피험자 평균) — 그 영역에 범주 정보가 있나
#    share         cat_cross − 우연 / cat_within − 우연 (공통 정도, cat_within이 우연보다 0.03 이상 높을 때만)
#    rr_within · rr_cross (11-19 추가)  커널 릿지(one-hot, λ 0.1) 범주 분류기: 자기 test / 다른 피험자 test — 패턴 상관보다 민감한 측정
#    rr_share    (rr_cross − 우연) / (rr_within − 우연), rr_within이 우연 + 0.03 이상일 때만 · rr_by_sub = 피험자별 rr_within
#  피험자별 차이: 영역 · 피험자마다 "다른 6명과의 평균 obj_gap" → 특정 피험자만 다른 영역이 있는지
# 사용 (실행 위치 $W/run): python diag_isc.py --data_root $DR --subs sub-0001,...,sub-0008 --out $W/prism_calib/isc.json
import os, sys, json, time, argparse, itertools, numpy as np, torch
import torch.nn.functional as F
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True); p.add_argument('--subs', required=True); p.add_argument('--out', required=True); p.add_argument('--min_px', type=int, default=40)
a = p.parse_args()
dev = 'cuda'; t0 = time.time(); SUBS = a.subs.split(','); FRAMES = list(range(2, 8))
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
has = lambda k_: all(os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')) for s in SUBS)
SPL = {'train': [k_ for k_ in lst('train') if has(k_)], 'test': [k_ for k_ in lst('test') if has(k_)]}
CL = sorted({k_.split('/')[0] for k_ in SPL['train']}); Y = {s: torch.tensor([CL.index(k_.split('/')[0]) for k_ in v], device=dev) for s, v in SPL.items()}
print('물체 ' + ' · '.join(f'{s} {len(v)}' for s, v in SPL.items()) + f' | 범주 {len(CL)} | 피험자 {len(SUBS)}', flush=True)
here = os.path.dirname(os.path.abspath(__file__))
lab = torch.as_tensor(np.load(os.path.join(here, 'assets', 'mmp_256_regions.npy'))).long().to(dev).flatten()
NAME = {int(k): v for k, v in json.load(open(os.path.join(here, 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
REG = {}
for r, n in NAME.items(): REG.setdefault(n.split('_', 1)[-1], []).append(r)
RM = {n: torch.isin(lab, torch.tensor(ids, device=dev)) for n, ids in REG.items()}; RM = {n: m for n, m in RM.items() if int(m.sum()) >= a.min_px}
RM['전체'] = lab > 0
print(f'영역 {len(RM) - 1}개 (픽셀 {a.min_px}개 이상) + 전체', flush=True)


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


@torch.no_grad()
def collect(split, s):
    o = []
    for i in range(0, len(SPL[split]), 32):
        x = norm_in(torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')).astype(np.float32)[FRAMES] for k_ in SPL[split][i:i + 32]])).to(dev))
        o.append(x.mean(1).flatten(1).half())
    return torch.cat(o)                                                    # [N, 65536]


X = {}
for s in SUBS:
    tr = collect('train', s).float(); te = collect('test', s).float(); mu = tr.mean(0, keepdim=True)
    X[s] = {'train': (tr - mu).half(), 'test': (te - mu).half()}; print(f'  {s} 끝 {time.time() - t0:.0f}s', flush=True)
PAIRS = list(itertools.permutations(range(len(SUBS)), 2)); yt = Y['test']; ytr = Y['train']
same_cat = (yt[:, None] == yt[None]); eye = torch.eye(len(yt), dtype=torch.bool, device=dev)
ch_obj = 1 / len(yt); ch_wc = float(np.mean([1 / int((yt == c).sum()) for c in yt])); ch_cat = 1 / len(CL)
zs = lambda t: F.normalize(t - t.mean(1, keepdim=True), dim=-1)            # 패턴 상관용 (물체마다 영역 안 평균 빼고 정규화)
R = {'chance': {'obj': round(ch_obj, 4), 'obj_wc': round(ch_wc, 4), 'cat': round(ch_cat, 4)}, 'subs': SUBS, 'regions': {}}
for n, m in RM.items():
    te = [X[s]['test'][:, m].float() for s in SUBS]; tr = [X[s]['train'][:, m].float() for s in SUBS]
    cen = [torch.stack([t[ytr == c].mean(0) for c in range(len(CL))]) for t in tr]                 # 범주 중심 [13, P]
    tez = [zs(t) for t in te]; cenz = [zs(c) for c in cen]
    tew = [zs(t - c[yt]) for t, c in zip(te, cen)]                                                   # 범주 평균 뺀 물체 고유 패턴
    top1, wc, gap, cc = [], [], [], []; per = np.zeros((len(SUBS), len(SUBS)))
    for i, j in PAIRS:
        Sm = tez[i] @ tez[j].t(); top1.append(float((Sm.argmax(1) == torch.arange(len(yt), device=dev)).float().mean()))
        g_ = float(Sm[eye].mean() - Sm[~eye].mean()); gap.append(g_); per[i, j] = g_
        Sw = (tew[i] @ tew[j].t()).masked_fill(~same_cat, -9); wc.append(float((Sw.argmax(1) == torch.arange(len(yt), device=dev)).float().mean()))
        cc.append(float(((tez[j] @ cenz[i].t()).argmax(1) == yt).float().mean()))
    cw = float(np.mean([float(((tez[i] @ cenz[i].t()).argmax(1) == yt).float().mean()) for i in range(len(SUBS))]))
    ntr = [F.normalize(t, dim=-1) for t in tr]; nte = [F.normalize(t, dim=-1) for t in te]; oh = F.one_hot(ytr, len(CL)).float(); rw = []; rc = []
    for i in range(len(SUBS)):
        Kt = ntr[i] @ ntr[i].t(); A = torch.linalg.solve(Kt + 0.1 * Kt.diagonal().mean() * torch.eye(len(Kt), device=dev), oh)
        for j in range(len(SUBS)):
            acc_ = float(((nte[j] @ ntr[i].t() @ A).argmax(1) == yt).float().mean()); (rw if i == j else rc).append(acc_)
    r = {'px': int(m.sum()), 'rr_within': round(float(np.mean(rw)), 4), 'rr_cross': round(float(np.mean(rc)), 4),
         'rr_share': round((float(np.mean(rc)) - ch_cat) / (float(np.mean(rw)) - ch_cat), 3) if float(np.mean(rw)) - ch_cat >= 0.03 else None, 'rr_by_sub': [round(x, 3) for x in rw], 'obj_top1': round(float(np.mean(top1)), 4), 'obj_top1_wc': round(float(np.mean(wc)), 4), 'obj_gap': round(float(np.mean(gap)), 4),
         'cat_cross': round(float(np.mean(cc)), 4), 'cat_within': round(cw, 4),
         'share': round((float(np.mean(cc)) - ch_cat) / (cw - ch_cat), 3) if cw - ch_cat >= 0.03 else None,
         'gap_by_sub': [round(float((per[i].sum() + per[:, i].sum()) / (2 * (len(SUBS) - 1))), 4) for i in range(len(SUBS))]}
    R['regions'][n] = r
json.dump(R, open(a.out, 'w'), ensure_ascii=False, indent=1)
rs = [(n, r) for n, r in R['regions'].items() if n != '전체']
print('전체', json.dumps(R['regions']['전체'], ensure_ascii=False))
for key, title in (('rr_within', '자기 범주 읽기(릿지) 상위'), ('rr_cross', '범주 옮겨 읽기(릿지) 상위'), ('obj_top1', '같은 물체 고르기 상위')):
    print(f'--- {title} 12 (우연 obj {ch_obj:.3f} · cat {ch_cat:.3f})')
    for n, r in sorted(rs, key=lambda t: -t[1][key])[:15]: print(f'  {n:8s} px {r["px"]:5d} rr_within {r["rr_within"]:.3f} rr_cross {r["rr_cross"]:.3f} share {r["rr_share"]} | obj {r["obj_top1"]:.3f} wc {r["obj_top1_wc"]:.3f} | 피험자별 {r["rr_by_sub"]}')
n_obj = sum(r['obj_top1'] > 3 * ch_obj for _, r in rs); n_cat = sum(r['rr_cross'] > ch_cat + 0.05 for _, r in rs); n_w = sum(r['rr_within'] > ch_cat + 0.05 for _, r in rs)
print(f'영역 {len(rs)}개 중: 같은 물체 고르기 > 우연 3배 {n_obj}개 · 범주 옮겨 읽기 > 우연 + 0.05 {n_cat}개 · 자기 범주 읽기 > 우연 + 0.05 {n_w}개')
print('ISC_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
