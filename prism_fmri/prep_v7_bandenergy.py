# V-7 (11-26 사용자 확인 12:32): Variational 정답 = 범주 평균을 뺀 공간 주파수 에너지 3개 (V-5 특징 band_lo · band_mid · band_hi)
#  입력: v5_feat.npz (keys [N] '<범주>/<uid>', feat [N, 10] 영상 10프레임 평균, names) — v5_feat.py
#  train 물체(annotations/core_train_list.txt)만으로 범주 평균 · 표준편차 → (값 − 정답 범주 train 평균) / train 표준편차
#  test 물체 값은 같은 통계로 바꿔 두지만 학습에는 안 쓰임 (train_prism은 train 목록 물체만 학습), 평가용
#  출력 npz: keys, shape [N, 6, 3] (6 = 학습 fMRI 프레임 수, 물체마다 같은 값), shape_ok — train_prism.py --var_heads shape --view_tgt <out>
# 사용: python prep_v7_bandenergy.py --feat $W/v5_feat.npz --data_root $DR --out $TGT/v7_bandenergy.npz
import os, argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--feat', required=True); p.add_argument('--data_root', required=True); p.add_argument('--out', required=True)
p.add_argument('--bands', default='band_lo,band_mid,band_hi'); p.add_argument('--nframe', type=int, default=6)
a = p.parse_args()
z = np.load(a.feat); KEYS = [str(k) for k in z['keys']]; NM = [str(n) for n in z['names']]
F = z['feat'][:, [NM.index(b) for b in a.bands.split(',')]].astype(np.float64); ok = np.isfinite(F).all(1)
TR = set(l.strip() for l in open(os.path.join(a.data_root, 'annotations', 'core_train_list.txt')) if l.strip())
TE = set(l.strip() for l in open(os.path.join(a.data_root, 'annotations', 'core_test_list.txt')) if l.strip())
itr = np.array([k in TR for k in KEYS]) & ok; cat = np.array([k.split('/')[0] for k in KEYS])
cm = {c: F[itr & (cat == c)].mean(0) for c in set(cat[itr])}
R = F - np.stack([cm.get(c, F[itr].mean(0)) for c in cat]); sd = R[itr].std(0) + 1e-8; R = np.nan_to_num(R / sd)
np.savez(a.out, keys=np.array(KEYS), shape=np.repeat(R[:, None], a.nframe, 1).astype(np.float32), shape_ok=ok,
         bands=np.array(a.bands.split(',')), train_sd=sd.astype(np.float32))
nte = sum(k in TE for k in KEYS)
print(f'[v7] 물체 {len(KEYS)} (정상 {int(ok.sum())}, train {int(itr.sum())}, test {nte}) | 범주 13개 train 통계 | 범주 뺀 train 표준편차 '
      + ' '.join(f'{b} {v:.4f}' for b, v in zip(a.bands.split(','), sd)) + f' → {a.out}', flush=True)
