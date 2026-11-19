# -*- coding: utf-8 -*-
# 11-19 진단: 같은 물체를 본 데이터로 피험자 B의 패턴을 피험자 A의 패턴으로 옮기는 선형 사상(기능 정렬)을 맞추면, A의 범주 분류기가 B에게 통하나 — 학습 없음(커널 릿지)
#  배경: 해부 정렬(평면 지도)만으로는 분류기를 옮기면 찍기 수준(diag_isc.py: 자기 0.448 · 옮김 0.111). Haxby 2011 · Guntupalli 2016은 같은 자극으로 맞춘 기능 정렬로 피험자 간 분류가 피험자 내 수준이 됨을 보고
#  절차 (쌍 A ← B): ① A의 train 물체로 범주 분류기(릿지 one-hot) ② 공유 물체 N개로 B → A 사상(릿지, 영역 픽셀 → 영역 픽셀) ③ B의 test 물체 패턴을 사상해 A의 분류기로 범주 맞히기
#  N = 정렬에 쓰는 공유 물체 수 (새 피험자에게서 얻어야 하는 "자극 – fMRI 짝" 수, 범주 라벨은 필요 없음). test 104개는 정렬에 안 씀
#  출력: 영역 묶음마다 [자기 / 옮김(정렬 없음) / 정렬 뒤 N별] 범주 정확도 (42쌍 평균, 찍기 1/13)
# 사용 (실행 위치 $W/run): python diag_align.py --data_root $DR --subs ... --out $W/prism_calib/align.json
import os, sys, json, time, argparse, itertools, numpy as np, torch
import torch.nn.functional as F
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True); p.add_argument('--subs', required=True); p.add_argument('--out', required=True); p.add_argument('--ns', default='50,100,300,600,1300')
a = p.parse_args()
dev = 'cuda'; t0 = time.time(); SUBS = a.subs.split(','); FRAMES = list(range(2, 8)); NS = len(SUBS)
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
has = lambda k_: all(os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')) for s in SUBS)
SPL = {'train': [k_ for k_ in lst('train') if has(k_)], 'test': [k_ for k_ in lst('test') if has(k_)]}
CL = sorted({k_.split('/')[0] for k_ in SPL['train']}); Y = {s: torch.tensor([CL.index(k_.split('/')[0]) for k_ in v], device=dev) for s, v in SPL.items()}
here = os.path.dirname(os.path.abspath(__file__))
lab = torch.as_tensor(np.load(os.path.join(here, 'assets', 'mmp_256_regions.npy'))).long().to(dev).flatten()
NAME = {int(k): v for k, v in json.load(open(os.path.join(here, 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
ids = lambda names: torch.tensor([r for r, n in NAME.items() if n.split('_', 1)[-1] in names], device=dev)
G = {'전체': lab > 0, '초기시각 V1-V4': torch.isin(lab, ids({'V1', 'V2', 'V3', 'V4'})), 'V1': torch.isin(lab, ids({'V1'})), 'V4': torch.isin(lab, ids({'V4'})),
     '배쪽·LO': torch.isin(lab, ids({'LO1', 'LO2', 'PIT', 'FFC', 'V8', 'VVC', 'PH', 'VMV1', 'VMV2', 'VMV3', 'TE2p', 'TF'})),
     '등쪽': torch.isin(lab, ids({'V3A', 'V3B', 'V6', 'V6A', 'V7', 'IPS1', 'MT', 'MST', 'LO3'}))}


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


@torch.no_grad()
def collect(split, s):
    o = []
    for i in range(0, len(SPL[split]), 32):
        x = norm_in(torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')).astype(np.float32)[FRAMES] for k_ in SPL[split][i:i + 32]])).to(dev))
        o.append(x.mean(1).flatten(1)[:, G['전체']].half())
    return torch.cat(o)


X = {}
for s in SUBS:
    tr = collect('train', s).float(); te = collect('test', s).float(); mu = tr.mean(0, keepdim=True); X[s] = {'train': (tr - mu).half(), 'test': (te - mu).half()}
print(f'읽기 끝 {time.time() - t0:.0f}s | train {len(SPL["train"])} · test {len(SPL["test"])} · 피험자 {NS}', flush=True)
sub_m = {g: m[G['전체']] for g, m in G.items()}; ytr, yte = Y['train'], Y['test']; oh = F.one_hot(ytr, len(CL)).double(); ntr = len(ytr)
def solve(K, T, lam): return torch.linalg.solve(K.double() + (lam * K.diagonal().mean().double() + 1e-6) * torch.eye(len(K), device=dev, dtype=torch.float64), T.double()).float()
NSH = [int(v) for v in a.ns.split(',')]; g_perm = torch.randperm(ntr, generator=torch.Generator().manual_seed(0)).to(dev)   # 정렬에 쓸 공유 물체 순서 (고정)
R = {'chance': round(1 / len(CL), 4), 'pairs': NS * (NS - 1), 'groups': {}}
for g, m in sub_m.items():
    tr = [F.normalize(X[s]['train'][:, m].float(), dim=-1) for s in SUBS]; te = [F.normalize(X[s]['test'][:, m].float(), dim=-1) for s in SUBS]
    cls = [solve(t @ t.t(), oh, 0.1) for t in tr]                                               # A의 분류기 (dual 계수)
    pred = lambda i, Z: (Z @ tr[i].t() @ cls[i]).argmax(1)                                       # Z [n, P] (A 공간) → 범주
    within = float(np.mean([float((pred(i, te[i]) == yte).float().mean()) for i in range(NS)]))
    raw = float(np.mean([float((pred(i, te[j]) == yte).float().mean()) for i in range(NS) for j in range(NS) if i != j]))
    r = {'px': int(m.sum()), 'within': round(within, 4), 'cross_noalign': round(raw, 4), 'aligned': {}}
    for n in NSH:
        idx = g_perm[:min(n, ntr)]; best = None
        for lam in (0.03, 0.3, 3.0):
            accs = []
            for i in range(NS):
                for j in range(NS):
                    if i == j: continue
                    Kb = tr[j][idx] @ tr[j][idx].t(); Wd = solve(Kb, tr[i][idx], lam)            # B → A (dual): [n, P_a]
                    Z = F.normalize(te[j] @ tr[j][idx].t() @ Wd, dim=-1); accs.append(float((pred(i, Z) == yte).float().mean()))
            v = float(np.mean(accs))
            if best is None or v > best[0]: best = (v, lam)
        r['aligned'][str(min(n, ntr))] = {'acc': round(best[0], 4), 'lam': best[1]}
    # (추가) ① 자기 데이터 N개로만 분류기를 학습했을 때 (정렬 N개와 같은 물체, 라벨 사용) — 정렬이 few-shot보다 나은지의 기준
    r['own_nshot'] = {}
    for n in NSH:
        idx = g_perm[:min(n, ntr)]; accs = []
        for i in range(NS):
            c_ = solve(tr[i][idx] @ tr[i][idx].t(), oh[idx], 0.1); accs.append(float(((te[i] @ tr[i][idx].t() @ c_).argmax(1) == yte).float().mean()))
        r['own_nshot'][str(min(n, ntr))] = round(float(np.mean(accs)), 4)
    # ② 다른 6명의 train 데이터를 A 공간으로 옮겨(5겹 밖 예측) A의 train과 합쳐 분류기 학습 → A의 test (데이터를 함께 쓰는 이득)
    if g in ('전체', '초기시각 V1-V4'):
        fold = torch.arange(ntr, device=dev) % 5; accs = []
        for i in range(NS):
            Zs = [tr[i]]
            for j in range(NS):
                if j == i: continue
                Zj = torch.zeros_like(tr[i])
                for f_ in range(5):
                    a_, b_ = fold != f_, fold == f_; Wd = solve(tr[j][a_] @ tr[j][a_].t(), tr[i][a_], 0.03); Zj[b_] = tr[j][b_] @ tr[j][a_].t() @ Wd
                Zs.append(F.normalize(Zj, dim=-1))
            Zall = torch.cat(Zs); c_ = solve(Zall @ Zall.t(), oh.repeat(NS, 1), 0.1); accs.append(float(((te[i] @ Zall.t() @ c_).argmax(1) == yte).float().mean()))
        r['pooled_aligned_7'] = round(float(np.mean(accs)), 4); r['pooled_aligned_7_by_sub'] = [round(x, 3) for x in accs]
    R['groups'][g] = r; print(f'[{g}]', json.dumps(r, ensure_ascii=False), flush=True)
json.dump(R, open(a.out, 'w'), ensure_ascii=False, indent=1); print('ALIGN_DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)
