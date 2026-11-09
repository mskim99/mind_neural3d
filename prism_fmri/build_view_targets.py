# V "변하는 정보" 정답 만들기 (11-09, CPU): 실루엣(prep_view_targets.py) · 시점 특징(trellis_tools.py dino --pool 1)을 "변하는 부분"으로 바꿔 한 npz로
#  이중 중심화: 물체마다 6프레임 평균을 빼고(물체 고유 = 고정 정보 제거), 프레임 번호마다 train 물체 평균을 뺌(모든 물체 공통 회전 효과 제거)
#  sil : [cx, cy, w, h, area] 이중 중심화 → train 표준편차로 나눔 → [N, 6, 5]
#  view: 프레임마다 DINO 요약 [54, 1024] 중 CLS(첫 토큰) + 7×7 격자 49 → 펼침(51,200) → 이중 중심화 → train으로 PCA → 상위 --pc개 → train 표준편차로 나눔 → [N, 6, pc]
#  통계(평균 · 표준편차 · PCA)는 train 물체만으로 (test 정보 안 씀). 프레임이 빠진 물체는 ok = False (학습 손실에서 뺌)
# 출력 <out> npz: keys, sil, sil_ok, (view, view_ok, view_evr = 설명 분산 비율) — train_prism.py --var_heads sil|view --view_tgt <out>
# 사용: python build_view_targets.py --sil <view_sil.npz> --data_root <fMRI-Shape> --out <npz> [--dino <폴더: dinof_v2 … dinof_v7의 상위> --pc 32]
import os, argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--sil', required=True); p.add_argument('--data_root', required=True); p.add_argument('--out', required=True)
p.add_argument('--dino', default=''); p.add_argument('--pc', type=int, default=32)
a = p.parse_args()
z = np.load(a.sil); KEYS = [str(k) for k in z['keys']]; FR = [int(f) for f in z['frames']]
TR = set(l.strip() for l in open(os.path.join(a.data_root, 'annotations', 'core_train_list.txt')) if l.strip())
itr = np.array([k in TR for k in KEYS])


def dcenter(Y, ok):                                       # Y [N, 6, D] → 이중 중심화 (train · ok 물체 통계)
    Y = Y - Y.mean(1, keepdims=True); m = itr & ok
    return Y - Y[m].mean(0, keepdims=True), m


R = {'keys': np.array(KEYS)}
S = z['sil'].astype(np.float64); sok = ~np.isnan(S).any((1, 2)); S = np.nan_to_num(S)
Sd, m = dcenter(S, sok); sd = Sd[m].reshape(-1, S.shape[-1]).std(0) + 1e-8
R['sil'] = (Sd / sd).astype(np.float32); R['sil_ok'] = sok
# shape (11-09): 범주 안 개체 형상 차이 — 물체 평균 실루엣 [cy, w, h, area](cx는 fMRI로 안 읽혀 뺌, V변화_전제확인.md) − train 범주 평균, train 표준편차로 나눔, 6프레임에 같은 값
cat = np.array([k.split('/')[0] for k in KEYS]); So = S.mean(1)[:, 1:]; cm = {c: So[itr & sok & (cat == c)].mean(0) for c in set(cat[itr & sok])}
Sr = So - np.stack([cm.get(c, So[itr & sok].mean(0)) for c in cat]); ssd = Sr[itr & sok].std(0) + 1e-8
R['shape'] = np.repeat((Sr / ssd)[:, None], S.shape[1], 1).astype(np.float32); R['shape_ok'] = sok
print(f'[shape] 범주 안 개체 차이 표준편차 ' + ' '.join(f'{n} {v:.4f}' for n, v in zip(['cy', 'w', 'h', 'area'], ssd)), flush=True)
print(f'[sil] 물체 {len(KEYS)} (정상 {int(sok.sum())}, train {int(m.sum())}) | 변하는 부분 표준편차 ' + ' '.join(f'{n} {v:.4f}' for n, v in zip(['cx', 'cy', 'w', 'h', 'area'], sd)), flush=True)
if a.dino:
    D = np.zeros((len(KEYS), len(FR), 50 * 1024), np.float32); vok = np.ones(len(KEYS), bool)
    for i, k in enumerate(KEYS):
        u = k.split('/')[1]
        for j, f in enumerate(FR):
            fp = f'{a.dino}/dinof_v{f}/{u}.npy'
            if not os.path.exists(fp): vok[i] = False; continue
            t = np.load(fp).astype(np.float32); D[i, j] = np.concatenate([t[:1], t[5:]], 0).ravel()   # CLS + 7×7 격자
        if (i + 1) % 300 == 0: print(f'[view] 읽음 {i + 1}/{len(KEYS)}', flush=True)
    Dd, m = dcenter(D, vok); X = Dd[m].reshape(-1, D.shape[-1]).astype(np.float64)
    G = X @ X.T; w, V = np.linalg.eigh(G); o = np.argsort(w)[::-1][:a.pc]; w, V = w[o], V[:, o]   # Gram → PCA (표본 수 < 차원)
    comp = (X.T @ V) / np.sqrt(np.clip(w, 1e-12, None))                                           # [51200, pc] 단위 주성분
    P = (Dd.reshape(-1, D.shape[-1]) @ comp).reshape(len(KEYS), len(FR), a.pc); psd = P[m].reshape(-1, a.pc).std(0) + 1e-8
    R['view'] = (P / psd).astype(np.float32); R['view_ok'] = vok; R['view_evr'] = (w / np.trace(G)).astype(np.float32)
    print(f'[view] 정상 {int(vok.sum())} | 상위 {a.pc}개 주성분 설명 분산 {float(w.sum() / np.trace(G)):.3f} (첫 5개 ' + ' '.join(f'{v:.3f}' for v in (w[:5] / np.trace(G))) + ')', flush=True)
np.savez(a.out, **R)
print('VIEW_TGT_DONE', a.out, {k: v.shape for k, v in R.items()}, flush=True)
