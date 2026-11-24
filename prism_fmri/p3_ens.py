# -*- coding: utf-8 -*-
# 11-24 P-3: 학습한 피험자 B의 test fMRI를 다른 학습 피험자 6명 쪽으로 각각 옮겨 그 사람들의 갈래에도 넣고, 범주 점수를 평균하면 B의 범주 정확도가 오르나
#  입력: align_infer.py --subs <7명> --fmri_subs <7명> --ns 1300 --save 1300,own --save_tok 0 의 출력 (own/<B>_as<B>_keep, n1300/<B>_as<A>_keep 의 components.npz cls_score)
#  점수는 물체마다 범주 13개에 대해 평균 0 · 표준편차 1로 맞춘 뒤 평균 (align_ens.py와 같음)
#  ① 자기 갈래만 ② 다른 6명 갈래 평균 ③ 자기 + 6명 평균, 넣는 갈래 수 k별(자기 포함 k개 / 자기 빼고 k개, 가능한 조합 전부의 평균), 물체 단위 맞고 틀림 바뀜
# 사용: python p3_ens.py <out 폴더> → <out>/p3.json
import os, sys, json, glob, itertools, numpy as np
O = sys.argv[1]; z = lambda s: (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-8)
own = {os.path.basename(d)[:8]: np.load(f'{d}/components.npz', allow_pickle=True) for d in sorted(glob.glob(f'{O}/own/*_keep'))}; SUBS = sorted(own); R = {'subs': {}, 'k': {}}
acc = lambda sc, y: float((sc.argmax(1) == y).mean()); KS, KO = {}, {}
for B in SUBS:
    y = own[B]['cls']; so = z(own[B]['cls_score']); assert list(own[B]['name']) == list(own[SUBS[0]]['name'])
    oth = {A: z(np.load(f'{O}/n1300/{B}_as{A[-2:]}_keep/components.npz', allow_pickle=True)['cls_score']) for A in SUBS if A != B}; L = list(oth.values())
    s6 = np.mean(L, 0); s7 = np.mean(L + [so], 0); h1, h6, h7 = so.argmax(1) == y, s6.argmax(1) == y, s7.argmax(1) == y
    R['subs'][B] = {'own': round(acc(so, y), 4), 'others6': round(acc(s6, y), 4), 'all7': round(acc(s7, y), 4), 'each_other': {A: round(acc(v, y), 4) for A, v in oth.items()},
                    'fixed': int((~h1 & h7).sum()), 'broken': int((h1 & ~h7).sum()), 'both_right': int((h1 & h7).sum()), 'both_wrong': int((~h1 & ~h7).sum()), 'hit_own': h1.astype(int).tolist(), 'hit_all7': h7.astype(int).tolist()}
    for k in range(1, 8):                                                  # 자기 포함 k개 (k = 1이 자기만)
        KS.setdefault(k, []).append(np.mean([acc(np.mean([so] + [L[i] for i in c], 0), y) for c in itertools.combinations(range(6), k - 1)]))
    for k in range(1, 7):                                                  # 자기 빼고 k개
        KO.setdefault(k, []).append(np.mean([acc(np.mean([L[i] for i in c], 0), y) for c in itertools.combinations(range(6), k)]))
    print(B, {k: R['subs'][B][k] for k in ('own', 'others6', 'all7', 'fixed', 'broken')}, flush=True)
m = lambda k: round(float(np.mean([R['subs'][B][k] for B in SUBS])), 4)
R['mean'] = {'own': m('own'), 'others6': m('others6'), 'all7': m('all7'), 'n_up': int(sum(R['subs'][B]['all7'] > R['subs'][B]['own'] for B in SUBS)), 'n_down': int(sum(R['subs'][B]['all7'] < R['subs'][B]['own'] for B in SUBS)),
             'fixed': int(sum(R['subs'][B]['fixed'] for B in SUBS)), 'broken': int(sum(R['subs'][B]['broken'] for B in SUBS))}
R['k'] = {'with_self': {str(k): round(float(np.mean(v)), 4) for k, v in KS.items()}, 'others_only': {str(k): round(float(np.mean(v)), 4) for k, v in KO.items()},
          'with_self_by_sub': {str(k): [round(float(x), 4) for x in v] for k, v in KS.items()}, 'subs': SUBS}
json.dump(R, open(f'{O}/p3.json', 'w'), ensure_ascii=False, indent=1); print('평균', R['mean']); print('k (자기 포함)', R['k']['with_self']); print('k (자기 빼고)', R['k']['others_only']); print('P3_ENS_DONE', f'{O}/p3.json')
