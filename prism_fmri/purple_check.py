# 보라 치우침 출처 확인 (11-08 색 실험 1 후속, 추론만, CPU)
#  색 실험 1에서 LN + 편차 ×s 토큰이 GT 색이 아니라 보라·분홍 쪽으로 진해짐 → 어디서 오는지 가림
#   single/M.npy        : train 정답 토큰 평균 M (LN 안 함)
#   single/lnM.npy      : LN(M)                         → 평균 물체 자체가 보라면 출처 = 정답 평균
#   single/Pbar.npy     : test 예측 평균 P̄ (LN 안 함)
#   single/lnPbar.npy   : LN(M + s·(P̄ − M))             → 예측 공통 치우침이 보라면 출처 = 모델 출력의 공통 방향
#   debias/<uid>.npy    : LN(M + s·(P − P̄))            → 공통 치우침을 뺀 개별 예측 (보라가 사라지는지)
#  P̄는 test 예측 104개 평균 (test 정답은 안 씀, 진단용). LN = 토큰마다 1024차원 affine 없는 layer_norm
#  출력 통계 <out>/purple_stats.json: 공통 치우침 비율 = ‖P̄ − M‖² / 평균 ‖P − M‖², 정답 쪽 같은 값(test 정답 평균)은 참고
# 사용: python purple_check.py --pred <예측 tokens 폴더> --tgt <shape_targets> --data_root <fMRI-Shape> --out <폴더> [--s 1.37]
import os, json, argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--pred', required=True); p.add_argument('--tgt', required=True); p.add_argument('--data_root', required=True)
p.add_argument('--out', required=True); p.add_argument('--s', type=float, default=1.37)
a = p.parse_args()
ids = lambda sp: [l.strip().split('/')[1] for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
tr = [u for u in ids('train') if os.path.exists(f'{a.tgt}/dino/{u}.npy')]; te = [u for u in ids('test') if os.path.exists(f'{a.pred}/{u}.npy')]
ld = lambda f: np.load(f).astype(np.float64)


def mean(fs):
    acc = None
    for f in fs:
        x = ld(f); acc = x if acc is None else acc + x
    return acc / len(fs)


def ln(t):
    return (t - t.mean(-1, keepdims=True)) / np.sqrt(t.var(-1, keepdims=True) + 1e-5)


M = mean([f'{a.tgt}/dino/{u}.npy' for u in tr]); Pb = mean([f'{a.pred}/{u}.npy' for u in te]); Gb = mean([f'{a.tgt}/dino/{u}.npy' for u in te])
print(f'train {len(tr)} test {len(te)} | 토큰 표준편차 M {M.std(-1).mean():.3f} · Pbar {Pb.std(-1).mean():.3f}', flush=True)
os.makedirs(f'{a.out}/single', exist_ok=True); os.makedirs(f'{a.out}/debias', exist_ok=True)
for n, t in (('M', M), ('lnM', ln(M)), ('Pbar', Pb), ('lnPbar', ln(M + a.s * (Pb - M)))):
    np.save(f'{a.out}/single/{n}.npy', t.astype(np.float16))
dp, dg = [], []
for u in te:
    P = ld(f'{a.pred}/{u}.npy'); dp.append(((P - M) ** 2).sum()); dg.append(((ld(f'{a.tgt}/dino/{u}.npy') - M) ** 2).sum())
    np.save(f'{a.out}/debias/{u}.npy', ln(M + a.s * (P - Pb)).astype(np.float16))
st = dict(common_ratio_pred=float(((Pb - M) ** 2).sum() / np.mean(dp)), common_ratio_gt=float(((Gb - M) ** 2).sum() / np.mean(dg)),
          cos_common_pred_gt=float(((Pb - M) * (Gb - M)).sum() / np.sqrt(((Pb - M) ** 2).sum() * ((Gb - M) ** 2).sum())), s=a.s)
print(f"PURPLE 공통 치우침 비율 예측 {st['common_ratio_pred']:.3f} · 정답 {st['common_ratio_gt']:.3f} · 두 공통 방향 cos {st['cos_common_pred_gt']:.3f}", flush=True)
json.dump(st, open(f'{a.out}/purple_stats.json', 'w'), indent=1); print('PURPLE_TOK_DONE', flush=True)
