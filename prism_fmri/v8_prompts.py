# V-8 (11-28): 외형 단계 문장 만들기 (로컬, 학습 없음). 기준 ㉠ = 기존 Ours 피험자 1번 문장 "a {예측 범주}" (prompt.json) 그대로에 수식어만 더함
#  ㉡ detail: high band(공간 주파수 48 ~ 160 에너지, v5_feat band_hi)에서 정답 범주 train 평균을 뺀 값 → test 104개 순위 위 1/3 "a highly detailed, intricate {명사}",
#            아래 1/3 "a plain, simple {명사}", 가운데는 기준 문장 그대로
#  ㉢ color: 물체 평균색(v9_feat lab_mean, 10프레임 평균, cv2 Lab) → 기본 색 11개 원형 중 Lab 거리 최소 → "a {색} {명사}"
#  값 출처 --src gt  : 자극에서 잰 정답 값 (상한 진단)
#         --src pred: 1단계 릿지 예측(<pred npz>: name · band_hi_res · lab_res) + 예측 범주(문장의 명사)의 train 평균 (정답 범주 안 씀)
# 출력: <out>/<src>_detail.json · <src>_color.json · <src>_values.json(물체별 값 · 고른 말)
import os, json, argparse, numpy as np, cv2
p = argparse.ArgumentParser()
p.add_argument('--base', required=True); p.add_argument('--v5', required=True); p.add_argument('--v9', default='')
p.add_argument('--ann', default=r'C:\Users\jinsw712\Desktop\Files\EEG\DataSet\MinD-3D++\fMRI-Shape\annotations')
p.add_argument('--src', default='gt', choices=['gt', 'pred']); p.add_argument('--pred', default=''); p.add_argument('--out', required=True)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
lst = lambda s: [l.strip() for l in open(os.path.join(a.ann, f'core_{s}_list.txt')) if l.strip()]
TR, TE = lst('train'), lst('test'); cat = lambda k: k.split('/')[0]; uid = lambda k: k.split('/')[-1]
BASE = json.load(open(a.base, encoding='utf-8'))
noun = {u: s.split(' ', 1)[1] for u, s in BASE.items()}
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text_prompts.py'), encoding='utf-8').read()
NAME = eval(_t[_t.index('NAME = ') + 7: _t.index('}', _t.index('NAME = ')) + 1])      # text_prompts.py의 범주 id → 이름 표 그대로
N2C = {n: c for c, n in NAME.items()}                                                    # 문장 명사 → 범주 id (예측 범주의 train 평균에 씀)
art = lambda w: 'an' if w[0] in 'aeiou' else 'a'
Z5 = np.load(a.v5); K5 = {str(k): i for i, k in enumerate(Z5['keys'])}; BH = Z5['feat'][:, list(Z5['names']).index('band_hi')].astype(np.float64)
cm_hi = {c: np.mean([BH[K5[k]] for k in TR if cat(k) == c]) for c in {cat(k) for k in TR}}
vals = {}
if a.v9:
    Z9 = np.load(a.v9); K9 = {str(k): i for i, k in enumerate(Z9['keys'])}; LAB = Z9['lab_mean'].astype(np.float64)
    cm_lab = {c: np.mean([LAB[K9[k]] for k in TR if cat(k) == c], 0) for c in cm_hi}
if a.src == 'gt':
    hi = {uid(k): BH[K5[k]] - cm_hi[cat(k)] for k in TE}
    lab = {uid(k): LAB[K9[k]] for k in TE} if a.v9 else {}
else:
    P = np.load(a.pred); names = [uid(str(x)) for x in P['name']]
    hi = dict(zip(names, P['band_hi_res'].astype(np.float64)))
    lab = {u: cm_lab[N2C[noun[u]]] + r for u, r in zip(names, P['lab_res'].astype(np.float64))} if 'lab_res' in P.files else {}
# 기본 색 원형 (sRGB) → cv2 Lab (v9_feat와 같은 변환: L* 0 ~ 100, a* · b* = 값 − 128)
PROTO = {'red': (200, 30, 30), 'orange': (230, 120, 20), 'yellow': (230, 210, 40), 'green': (40, 150, 50), 'blue': (30, 70, 200), 'purple': (120, 40, 160),
         'pink': (230, 120, 170), 'brown': (120, 70, 30), 'black': (25, 25, 25), 'white': (240, 240, 240), 'gray': (128, 128, 128)}
def tolab(rgb):
    l = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64); return np.array([l[0] * 100 / 255, l[1] - 128, l[2] - 128])
PL = {n: tolab(c) for n, c in PROTO.items()}
cname = lambda v: min(PL, key=lambda n: np.linalg.norm(PL[n] - v))
us = [uid(k) for k in TE if uid(k) in BASE]
r = np.argsort(np.argsort([hi[u] for u in us])); n3 = len(us) / 3
det, col = {}, {}
for u, rk in zip(us, r):
    w = 'high' if rk >= len(us) - n3 else ('low' if rk < n3 else 'mid')
    det[u] = f'a highly detailed, intricate {noun[u]}' if w == 'high' else (f'a plain, simple {noun[u]}' if w == 'low' else BASE[u])
    v = {'noun': noun[u], 'hi_res': round(float(hi[u]), 4), 'detail': w}
    if lab:
        c = cname(lab[u]); col[u] = f'{art(c)} {c} {noun[u]}'; v.update(lab=[round(float(x), 2) for x in lab[u]], color=c)
    vals[u] = v
json.dump(det, open(f'{a.out}/{a.src}_detail.json', 'w'), indent=1)
if col: json.dump(col, open(f'{a.out}/{a.src}_color.json', 'w'), indent=1)
json.dump(vals, open(f'{a.out}/{a.src}_values.json', 'w'), indent=1)
from collections import Counter
print(a.src, '| detail', Counter(v['detail'] for v in vals.values()), '| color', Counter(v.get('color') for v in vals.values()), flush=True)
