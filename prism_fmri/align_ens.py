# -*- coding: utf-8 -*-
# 11-23: align_infer.py --save 결과(components.npz)를 모아, 새 피험자를 학습 피험자 여럿에게 각각 옮겨 넣은 범주 점수를 평균했을 때의 정확도
#  (한 명에게 옮긴 값의 평균 = align.json의 mean, 여기서는 "여럿의 점수를 평균한 뒤 고르기")
# 사용: python align_ens.py <out 폴더 (설정별 하위 폴더)> → <out>/summary.json
import os, sys, json, glob, numpy as np
O = sys.argv[1]; R = {}
for d in sorted(glob.glob(f'{O}/*/')):
    nm = os.path.basename(d.rstrip('/'))
    if not os.path.exists(f'{d}/align.json'): continue
    J = json.load(open(f'{d}/align.json', encoding='utf-8')); r = {'own_mean': J.get('own_mean'), 'single_mean': J.get('mean', {}).get('all'), 'single_by_A': {k: {n: v['test'] for n, v in p['n'].items()} for k, p in J['pairs'].items()}, 'ens': {}}
    for nd in sorted(glob.glob(f'{d}/n*/'), key=lambda x: int(os.path.basename(x.rstrip('/'))[1:])):
        C = [np.load(f) for f in sorted(glob.glob(f'{nd}/*_keep/components.npz'))]
        if not C: continue
        z = lambda s: (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-8)
        sc = np.mean([z(c['cls_score']) for c in C], 0); r['ens'][os.path.basename(nd.rstrip('/'))[1:]] = {'acc': round(float((sc.argmax(1) == C[0]['cls']).mean()), 4), 'k': len(C)}
    R[nm] = r; print(nm, '| 본인', r['own_mean'], '| 한 명에게', r['single_mean'], '| 점수 평균', {k: v['acc'] for k, v in r['ens'].items()}, flush=True)
json.dump(R, open(f'{O}/summary.json', 'w'), ensure_ascii=False, indent=1); print('ENS_DONE', f'{O}/summary.json')
