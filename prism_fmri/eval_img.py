# A안 2단계 평가 (11-10, CPU): 예측 이미지 임베딩(components.npz 'img') vs 정답(자극 프레임 0 unCLIP 인코더 임베딩, <clipimg>/<uid>.npy)
#  test 물체 안 검색: top-1 · top-5 · 2-way(정답 cos > 다른 물체 cos 비율), 정답과 cos
#  기준선: 범주 평균(train 정답 임베딩을 범주별 평균) — 정답 범주 / 예측 범주(Semantic top-1). 예측이 범주 이상을 담으면 기준선보다 높아야 함
# 사용: python eval_img.py --comp <components.npz> --clipimg <폴더> --train_list <core_train_list.txt> [--json <저장>]
import os, json, argparse, numpy as np
p = argparse.ArgumentParser(); p.add_argument('--comp', required=True); p.add_argument('--clipimg', required=True)
p.add_argument('--train_list', required=True); p.add_argument('--json', default=''); a = p.parse_args()
z = np.load(a.comp); names = [str(k) for k in z['name']]; classes = [str(c) for c in z['classes']]
nrm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
G = nrm(np.stack([np.load(f"{a.clipimg}/{k.split('/')[1]}.npy") for k in names]))
tr = [l.strip() for l in open(a.train_list) if l.strip()]
T = {c: nrm(np.stack([np.load(f"{a.clipimg}/{k.split('/')[1]}.npy") for k in tr if k.startswith(c + '/') and os.path.exists(f"{a.clipimg}/{k.split('/')[1]}.npy")]).mean(0)) for c in classes}
def metrics(P):
    S = P @ G.T; d = np.diag(S); n = len(d)
    rank = (S > d[:, None]).sum(1)
    return {'cos': float(d.mean()), 'top1': float((rank == 0).mean()), 'top5': float((rank < 5).mean()),
            '2way': float(((S < d[:, None]).sum(1) / (n - 1)).mean())}
R = {}
if 'img' in z: R['pred'] = metrics(nrm(z['img']))
R['class_mean_true'] = metrics(np.stack([T[classes[c]] for c in z['cls']]))
R['class_mean_pred'] = metrics(np.stack([T[classes[c]] for c in z['cls_pred']]))
R['chance'] = {'top1': 1 / len(names), 'top5': 5 / len(names), '2way': 0.5}
for k, v in R.items(): print(f'{k:>16} ' + ' '.join(f'{m} {x:.3f}' for m, x in v.items()), flush=True)
if a.json: json.dump(R, open(a.json, 'w'), indent=1)
print('EVAL_IMG_DONE', flush=True)
