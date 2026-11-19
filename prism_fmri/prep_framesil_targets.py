# -*- coding: utf-8 -*-
# 11-20 Variational 정답 = 프레임별 실루엣 차이 지도 (CPU, 학습 없음) — 사용자 11-19: V = 프레임별 차이(해석 가능한 정답), 범주 · 물체 차이는 V에 넣지 않음
#  입력 silhouettes32.npz (자극 영상 프레임 round(19.2k) 10장의 실루엣 32×32, 값 0~255 = 덮인 비율)
#  정답 [N, 6, g·g]: fMRI 프레임 2~7에 대응하는 영상 프레임 k의 실루엣(g×g로 면적 평균) − 그 물체의 6프레임 평균(물체 · 범주 고유 모양 제거)
#                    − 프레임 번호별 train 물체 평균(모든 물체에 공통인 회전 효과 제거), train 전체 표준편차 하나로 나눔(화소별로 안 나눠 지도 모양이 그대로 남음)
#  통계는 train 물체만으로. 출력 npz: keys(uid) · sil [N, 6, g·g] · sil_ok · sd · frame_mean — train_prism.py --var_heads sil --view_tgt <out>
#  손실 L_vt = 표준화된 값의 MSE → 0만 내면 약 1.0, 1보다 내려간 만큼이 fMRI에서 읽힌 프레임 차이
# 사용: python prep_framesil_targets.py --sil $TGT/silhouettes32.npz --data_root $DR --out $TGT/framesil16.npz [--grid 16]
import os, argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--sil', required=True); p.add_argument('--data_root', required=True); p.add_argument('--out', required=True); p.add_argument('--grid', type=int, default=16)
a = p.parse_args()
FRAMES = list(range(2, 8))                                # train_prism.py FRAMES와 같음
z = np.load(a.sil); U = [str(u) for u in z['uids']]; S = z['sil'].astype(np.float32)[:, FRAMES] / 255.0; N = len(U); g = a.grid; k = 32 // g
S = S.reshape(N, len(FRAMES), g, k, g, k).mean((3, 5)).reshape(N, len(FRAMES), g * g)
TR = {l.strip().split('/')[1] for l in open(os.path.join(a.data_root, 'annotations', 'core_train_list.txt')) if l.strip()}
itr = np.array([u in TR for u in U]); assert itr.sum() > 100, 'train 물체가 안 잡힘'
D = S - S.mean(1, keepdims=True); fm = D[itr].mean(0, keepdims=True); D = D - fm; sd = float(D[itr].std())
np.savez_compressed(a.out, keys=np.array(U), sil=(D / sd).astype(np.float32), sil_ok=np.ones(N, bool), sd=sd, frame_mean=fm[0].astype(np.float32), grid=g)
print(f'물체 {N} (train {int(itr.sum())}) · 정답 {D.shape} · 표준편차 {sd:.4f} · 공통 회전 효과가 차지한 분산 비율 {float((fm ** 2).mean() / ((D + fm)[itr] ** 2).mean()):.3f} → {a.out}')
