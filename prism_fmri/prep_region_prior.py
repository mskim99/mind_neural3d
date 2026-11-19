# -*- coding: utf-8 -*-
# 11-20 (b) 영역별 "피험자 간 공통성" 사전값 → prism.RegionWeight의 공통 가중 초기값 (train_prism.py --region_prior)
#  diag_overlap.py의 범주 단위 피험자 간 상관(ISC)과 같은 식인데 **train 물체만** 씀 (test · 검증 물체(범주마다 목록 끝 holdout개)는 뺌 → 평가 물체 정보가 학습에 안 들어감)
#    피험자마다 fMRI 프레임 2~7 평균(in_norm) → 픽셀마다 물체 축 z-score → 범주 평균 지도 [13, P] → 픽셀마다 피험자 쌍 상관(범주 축) 평균 → 영역(HCP-MMP id, 좌우 따로) 평균
#  영역 id 0(영역 없음)과 픽셀 min_px개 미만 영역은 NaN (RegionWeight가 가운데 값으로 둠)
#  피험자: 기본 7명 전부 (학습 피험자가 2명이어도 "사람 사이에 공통인 영역"은 7명으로 잼, train 물체만이라 평가 누수 없음)
#  출력 npz: prior [R] (영역별 범주 ISC), px [R], isc_cat [256, 256], subs, classes, n_obj · 요약 json 같은 이름 .json
# 사용 (실행 위치 $W/run, CPU): python prep_region_prior.py --data_root $DR --out $W/prism_calib/region_prior.npz
import os, json, time, argparse, itertools, numpy as np, torch
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True); p.add_argument('--out', required=True)
p.add_argument('--subs', default='sub-0001,sub-0002,sub-0003,sub-0005,sub-0006,sub-0007,sub-0008')
p.add_argument('--lr_avg', type=int, default=1)   # 11-20: 좌우 같은 이름 영역(L_x · R_x)의 값을 평균 (영역 하나 값이 잡음에 흔들리지 않게, 사용자 승인)
p.add_argument('--holdout', type=int, default=8); p.add_argument('--min_px', type=int, default=40); p.add_argument('--dev', default='cpu')
a = p.parse_args()
t0 = time.time(); SUBS = a.subs.split(','); FRAMES = list(range(2, 8)); dev = a.dev
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
has = lambda k_: all(os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')) for s in SUBS)
tr = lst('train'); by = {}
for k_ in tr: by.setdefault(k_.split('/')[0], []).append(k_)
H = {k_ for v in by.values() for k_ in v[-a.holdout:]} if a.holdout > 0 else set()
TEST = set(lst('test'))
KEYS = [k_ for k_ in tr if k_ not in H and k_ not in TEST and has(k_)]
CL = sorted({k_.split('/')[0] for k_ in KEYS}); Y = torch.tensor([CL.index(k_.split('/')[0]) for k_ in KEYS])
here = os.path.dirname(os.path.abspath(__file__))
lab = torch.as_tensor(np.load(os.path.join(here, 'assets', 'mmp_256_regions.npy'))).long().flatten(); M = lab > 0; P = int(M.sum()); R = int(lab.max()) + 1
print(f'train 물체 {len(KEYS)} (검증 {len(H)} · test 뺌) · 범주 {len(CL)} · 피험자 {len(SUBS)} · 픽셀 {P}', flush=True)


def norm_in(x):
    m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
    sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; return (x - mu) / sd * m


@torch.no_grad()
def collect(s):
    o = []
    for i in range(0, len(KEYS), 64):
        x = norm_in(torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')).astype(np.float32)[FRAMES] for k_ in KEYS[i:i + 64]])).to(dev))
        o.append(x.mean(1).flatten(1)[:, M.to(dev)])
    x = torch.cat(o); return (x - x.mean(0)) / x.std(0).clamp_min(1e-6)   # [N, P] 픽셀마다 물체 축 z-score


cat_m = []
for s in SUBS:
    z = collect(s); cat_m.append(torch.stack([z[Y.to(dev) == c].mean(0) for c in range(len(CL))])); print(f'  {s} 끝 {time.time() - t0:.0f}s', flush=True)
cat_m = torch.stack(cat_m)                                                   # [S, 13, P]
Z = cat_m - cat_m.mean(1, keepdim=True); Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-8)
isc = torch.stack([(Z[i] * Z[j]).sum(0) for i, j in itertools.combinations(range(len(SUBS)), 2)]).mean(0).cpu()   # [P] 범주 단위 피험자 간 상관
labm = lab[M]; prior = np.full(R, np.nan, np.float32); px = np.zeros(R, np.int64)
for r in range(1, R):
    m = labm == r; px[r] = int(m.sum())
    if px[r] >= a.min_px: prior[r] = float(isc[m].mean())
NAME0 = {int(k): v for k, v in json.load(open(os.path.join(here, 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
raw = prior.copy()
if a.lr_avg:                                                                  # 좌우 평균: 두 쪽 다 있으면 평균, 한쪽만 있으면 그 값
    for r in range(1, R):
        nm = NAME0.get(r, ''); tw = [q for q, n2 in NAME0.items() if q > 0 and n2.split('_', 1)[-1] == nm.split('_', 1)[-1] and np.isfinite(raw[q])]
        if np.isfinite(raw[r]) and tw: prior[r] = float(np.mean([raw[q] for q in tw]))
full = np.zeros(256 * 256, np.float32); full[M.numpy()] = isc.numpy()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(a.out, prior=prior, prior_raw=raw, lr_avg=a.lr_avg, px=px, isc_cat=full.reshape(256, 256), subs=np.array(SUBS), classes=np.array(CL), n_obj=len(KEYS))
NAME = {int(k): v for k, v in json.load(open(os.path.join(here, 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
ok = np.isfinite(prior); order = np.argsort(-np.where(ok, prior, -9))
top = [(NAME.get(int(r), str(r)), round(float(prior[r]), 3), int(px[r])) for r in order[:15]]
bot = [(NAME.get(int(r), str(r)), round(float(prior[r]), 3), int(px[r])) for r in order[:int(ok.sum())][-10:]]
summ = {'n_obj': len(KEYS), 'subs': SUBS, 'pixel_mean_isc': round(float(isc.mean()), 4), 'n_regions': int(ok.sum()), 'top15': top, 'bottom10': bot}
json.dump(summ, open(os.path.splitext(a.out)[0] + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('공통성 높은 영역:', ' · '.join(f'{n} {v}' for n, v, _ in top[:10]))
print('낮은 영역:', ' · '.join(f'{n} {v}' for n, v, _ in bot))
print(f'REGION_PRIOR_DONE {a.out} 영역 {int(ok.sum())}개 {time.time() - t0:.0f}s', flush=True)
