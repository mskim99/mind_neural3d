# V-8 ㉥ (11-28): 속도 차이 조향의 α 파일 만들기 (로컬)
#  plus = ㉡의 세부 많음 문장 "a highly detailed, intricate {명사}", minus = "a plain, simple {명사}", 기준 문장 = ㉠ 그대로 ("a {예측 범주}")
#  --mode fixed: --names의 물체마다 α 하나(--alpha) — 작은 묶음에서 α 크기 고르기
#  --mode z    : α = k × clip(z, ±2), z = high band(band_hi)의 범주 안 z 점수 = (값 − 정답 범주 train 평균) / 범주 뺀 train 표준편차
#                --src gt = 자극 값, --src pred = V-9 피험자 1번 예측(pred_v5 6번째 = band_hi, 범주 뺀 값)
# 출력: <out> json {이름: {plus, minus, alpha, z}}
import os, json, argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--base', required=True); p.add_argument('--v5', required=True); p.add_argument('--ann', default=r'C:\Users\jinsw712\Desktop\Files\EEG\DataSet\MinD-3D++\fMRI-Shape\annotations')
p.add_argument('--mode', choices=['fixed', 'z'], required=True); p.add_argument('--names', default=''); p.add_argument('--alpha', type=float, default=0)
p.add_argument('--src', default='gt', choices=['gt', 'pred']); p.add_argument('--pred', default=''); p.add_argument('--k', type=float, default=1); p.add_argument('--plus', default='a highly detailed, intricate {n}'); p.add_argument('--minus', default='a plain, simple {n}'); p.add_argument('--out', required=True)
a = p.parse_args()
lst = lambda s: [l.strip() for l in open(os.path.join(a.ann, f'core_{s}_list.txt')) if l.strip()]
TR, TE = lst('train'), lst('test'); cat = lambda k: k.split('/')[0]; uid = lambda k: k.split('/')[-1]
BASE = json.load(open(a.base, encoding='utf-8')); noun = {u: s.split(' ', 1)[1] for u, s in BASE.items()}
Z5 = np.load(a.v5); K5 = {str(k): i for i, k in enumerate(Z5['keys'])}; BH = Z5['feat'][:, list(Z5['names']).index('band_hi')].astype(np.float64)
cm = {c: np.mean([BH[K5[k]] for k in TR if cat(k) == c]) for c in {cat(k) for k in TR}}
sd = np.std([BH[K5[k]] - cm[cat(k)] for k in TR])
if a.src == 'gt': res = {uid(k): BH[K5[k]] - cm[cat(k)] for k in TE}
else:
    P = np.load(a.pred); res = {uid(str(k)): float(v) for k, v in zip(P['name'], P['pred_v5'][:, 5])}
z = {u: r / sd for u, r in res.items()}
names = [l.strip() for l in open(a.names) if l.strip()] if a.names else list(BASE)
out = {}
for u in names:
    al = a.alpha if a.mode == 'fixed' else a.k * float(np.clip(z[u], -2, 2))
    out[u] = {'plus': a.plus.format(n=noun[u], an=BASE[u]), 'minus': a.minus.format(n=noun[u], an=BASE[u]), 'alpha': round(al, 4), 'z': round(float(z[u]), 4)}
json.dump(out, open(a.out, 'w'), indent=1)
al = np.array([v['alpha'] for v in out.values()]); print(a.mode, a.src, len(out), 'α 범위', al.min().round(2), al.max().round(2), '| train 범주 뺀 sd', round(float(sd), 4), flush=True)
